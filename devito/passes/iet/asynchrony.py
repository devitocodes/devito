from collections import namedtuple
from functools import singledispatch
from ctypes import c_int

import cgen as c

from devito.ir import (AsyncCall, AsyncCallable, BlankLine, Call, Callable,
                       Conditional, DummyEq, DummyExpr, While, Increment, Iteration,
                       List, PointerCast, Return, FindNodes, FindSymbols,
                       ThreadCallable, EntryFunction, Transformer, make_callable,
                       maybe_alias)
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, CondNe, FieldFromComposite, FieldFromPointer,
                              Null)
from devito.tools import split
from devito.types import (Lock, Pointer, PThreadArray, QueueID, SharedData, Temp,
                          VolatileInt)

__all__ = ['pthreadify']


def pthreadify(graph, **kwargs):
    if not any(isinstance(i, AsyncCallable) for i in graph.efuncs.values()):
        return

    # A key function to help identify all non-constant Symbols in the IET
    defines = FindSymbols('defines').visit(graph.root.body)
    key = lambda i: i in defines and i.is_Symbol

    lower_async_objs(graph, key=key, tracker={}, **kwargs)

    # We need this one here to initialize the newly introduced objects, such as
    # SharedData and PThreadArray, as well as the packed/unpacked symbols
    DataManager(**kwargs).place_definitions(graph)


AsyncMeta = namedtuple('AsyncMeta', 'sdata threads init shutdown')


@iet_pass
def lower_async_objs(iet, **kwargs):
    # Different actions depending on the Callable type
    iet, efuncs = _lower_async_objs(iet, **kwargs)

    metadata = {'includes': ['pthread.h'],
                'efuncs': efuncs}

    if not isinstance(iet, EntryFunction):
        return iet, metadata

    iet, efuncs = inject_async_tear_updown(iet, **kwargs)
    metadata['efuncs'].extend(efuncs)

    return iet, metadata


@singledispatch
def _lower_async_objs(iet, tracker=None, sregistry=None, **kwargs):
    # All Callables, except for AsyncCallables, may containg one or more
    # AsyncCalls, which we have to lower into thread-activation code
    efuncs = []
    subs = {}
    for n in FindNodes(AsyncCall).visit(iet):
        # The efuncs in a Graph are processed bottom-up, so we can safely assume
        # that the `AsyncCallable` corresponding to `n` has already been processed
        assert n.name in tracker
        sdata = tracker[n.name].sdata

        if sdata.size == 1:
            d = sdata.index
            condition = CondNe(FieldFromComposite(sdata.symbolic_flag, sdata[d]), 1)
            activation = [While(condition)]
        else:
            d = Temp(name=sregistry.make_name(prefix=sdata.index.name))
            condition = CondNe(FieldFromComposite(sdata.symbolic_flag, sdata[d]), 1)
            activation = [DummyExpr(d, 0),
                          While(condition, DummyExpr(d, (d + 1) % sdata.size))]
        arguments = []
        for i in sdata.ncfields:
            for a in n.arguments:
                if maybe_alias(a, i):
                    arguments.append(a)
                    break
            else:
                arguments.append(i)
        activation.extend([DummyExpr(FieldFromComposite(i.base, sdata[d]), i)
                           for i in arguments])
        activation.append(
            DummyExpr(FieldFromComposite(sdata.symbolic_flag, sdata[d]), 2)
        )
        name = sregistry.make_name(prefix='activate')
        efunc = make_callable(name, activation)

        efuncs.append(efunc)
        subs[n] = Call(name, efunc.parameters)

    iet = Transformer(subs).visit(iet)

    return iet, efuncs


@_lower_async_objs.register(AsyncCallable)
def _(iet, key=None, tracker=None, sregistry=None, **kwargs):
    # Determine the max number of threads that can run this `iet` in parallel
    locks = [i for i in iet.parameters if isinstance(i, Lock)]
    npthreads = min([i.size for i in locks], default=1)
    if npthreads > 1:
        npthreads = sregistry.make_npthreads(npthreads)

    # The `cfields` are the constant fields, that is the fields whose value
    # definitely never changes across different executions of `ìet`; the
    # `ncfields` are instead the non-constant fields, that is the fields whose
    # value may or may not change across different calls to `iet`. Clearly objects
    # passed by pointer don't really matter
    ncfields, cfields = split(iet.parameters, key)

    # Postprocess `ncfields`
    ncfields = sanitize_ncfields(ncfields)

    # SharedData -- that is the data structure that will be used by the
    # main thread to pass information down to the child thread(s)
    sdata = SharedData(
        name=sregistry.make_name(prefix='sdata'),
        npthreads=npthreads,
        cfields=cfields,
        ncfields=ncfields,
        pname=sregistry.make_name(prefix='tsdata')
    )
    sbase = sdata.indexed

    # PThreadArray -- that is he pthreads that will execute the AsyncCallable
    # asynchronously
    threads = PThreadArray(
        name=sregistry.make_name(prefix='threads'),
        npthreads=npthreads
    )
    tbase = threads.indexed

    # Prepend the SharedData fields available upon thread activation
    preactions = [DummyExpr(i, FieldFromPointer(i.base, sbase)) for i in ncfields]
    preactions.append(BlankLine)

    # Append the flag reset
    postactions = [List(body=[
        BlankLine,
        DummyExpr(FieldFromPointer(sdata.symbolic_flag, sbase), 1)
    ])]

    wrap = List(body=preactions + list(iet.body.body) + postactions)

    # The thread has work to do when it receives the signal that all locks have
    # been set to 0 by the main thread
    wrap = Conditional(
        CondEq(FieldFromPointer(sdata.symbolic_flag, sbase), 2),
        wrap
    )

    # The thread keeps spinning until the alive flag is set to 0 by the main thread
    wrap = While(CondNe(FieldFromPointer(sdata.symbolic_flag, sbase), 0), wrap)

    # pthread functions expect exactly one argument of type void*
    tparameter = Pointer(name='_%s' % sdata.name)

    # Unpack `sdata`
    unpacks = [PointerCast(sdata, tparameter), BlankLine]
    for i in cfields:
        if i.is_AbstractFunction:
            unpacks.append(
                DummyExpr(i._C_symbol, FieldFromPointer(i._C_symbol, sbase))
            )
        else:
            unpacks.append(DummyExpr(i, FieldFromPointer(i.base, sbase), init=True))

    body = iet.body._rebuild(body=[wrap, Return(Null)], unpacks=unpacks)
    iet = ThreadCallable(iet.name, body, tparameter)

    d = sdata.index
    if sdata.size == 1:
        callback = lambda body: body
    else:
        footer = [BlankLine,
                  Increment(DummyEq(sbase, 1))]
        callback = lambda body: Iteration(list(body) + footer, d, threads.size - 1)

    # Create an efunc to initialize `sdata` and tear up the pthreads
    name = 'init_%s' % sdata.name
    body = []
    for i in sdata.cfields:
        if i.is_AbstractFunction:
            body.append(
                DummyExpr(FieldFromPointer(i._C_symbol, sbase), i._C_symbol)
            )
        elif isinstance(i, QueueID):
            body.append(DummyExpr(FieldFromPointer(i, sbase), i + d))
        else:
            body.append(DummyExpr(FieldFromPointer(i.base, sbase), i.base))
    body.extend([
        BlankLine,
        DummyExpr(FieldFromPointer(sdata.symbolic_flag, sbase), 1)
    ])
    body.extend([
        BlankLine,
        Call('pthread_create', (
            tbase + d, Null, Call(iet.name, [], is_indirect=True), sbase
        ))
    ])
    body = callback(body)
    parameters = sdata.cfields + (sdata, threads)
    init = Callable(name, body, 'void', parameters, 'static')

    # Create an efunc to shutdown the pthreads
    name = sregistry.make_name(prefix='shutdown')
    body = List(body=callback([
        While(CondEq(FieldFromPointer(sdata.symbolic_flag, sbase), 2)),
        DummyExpr(FieldFromPointer(sdata.symbolic_flag, sbase), 0),
        Call('pthread_join', (threads[d], Null))
    ]))
    shutdown = make_callable(name, body)

    # Track all the newly created objects
    tracker[iet.name] = AsyncMeta(sdata, threads, init, shutdown)

    return iet, []


def inject_async_tear_updown(iet, tracker=None, **kwargs):
    # Number of allocated asynchronous queues so far
    nqueues = 1

    tearup = []
    teardown = []
    for sdata, threads, init, shutdown in tracker.values():
        # Tear-up
        arguments = list(init.parameters)
        for n, a in enumerate(list(arguments)):
            if isinstance(a, QueueID):
                # Different pthreads use different queues
                arguments[n] = nqueues
        tearup.append(Call(init.name, arguments))
        nqueues += threads.size  # Update the next available queue ID

        # Tear-down
        arguments = list(shutdown.parameters)
        teardown.append(Call(shutdown.name, arguments))

    # Inject tearup and teardown
    tearup = List(
        header=c.Comment("Fire up and initialize pthreads"),
        body=tearup + [BlankLine]
    )
    teardown = List(
        header=c.Comment("Wait for completion of pthreads"),
        body=teardown
    )
    body = iet.body._rebuild(
        body=[tearup] + list(iet.body.body) + [BlankLine, teardown]
    )
    iet = iet._rebuild(body=body)

    efuncs = []
    efuncs.extend(i.init for i in tracker.values())
    efuncs.extend(i.shutdown for i in tracker.values())

    return iet, efuncs


# *** Utils

def sanitize_ncfields(ncfields):
    # Due to a bug in the NVC compiler (v<=22.7 and potentially later),
    # we have to use C's `volatile` more extensively than strictly necessary
    # to avoid flaky optimizations that would cause fauly behaviour in rare,
    # non-deterministic scenarios
    sanitized = []
    for i in ncfields:
        if i._C_ctype is c_int:
            sanitized.append(VolatileInt(name=i.name))
        else:
            sanitized.append(i)

    return sanitized

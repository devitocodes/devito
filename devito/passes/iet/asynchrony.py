from collections import OrderedDict
from ctypes import c_int

import cgen as c

from devito.ir import (AsyncCall, AsyncCallable, BlankLine, Call, Callable,
                       Conditional, DummyExpr, FindNodes, FindSymbols,
                       Iteration, List, PointerCast, Return, ThreadCallable,
                       Transformer, While, make_callable, maybe_alias)
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, CondNe, FieldFromComposite, FieldFromPointer,
                              Null)
from devito.tools import split
from devito.types import (Lock, Pointer, PThreadArray, QueueID, SharedData, Temp,
                          VolatileInt)

__all__ = ['pthreadify']


def pthreadify(graph, **kwargs):
    lower_async_callables(graph, root=graph.root, **kwargs)

    track = {i.name: i.sdata for i in graph.efuncs.values()
             if isinstance(i, ThreadCallable)}
    lower_async_calls(graph, track=track, **kwargs)

    DataManager(**kwargs).place_definitions(graph)


@iet_pass
def lower_async_callables(iet, root=None, sregistry=None):
    if not isinstance(iet, AsyncCallable):
        return iet, {}

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
    fields = iet.parameters
    defines = FindSymbols('defines').visit(root.body)
    ncfields, cfields = split(fields, lambda i: i in defines and i.is_Symbol)

    # Postprocess `ncfields`
    ncfields = sanitize_ncfields(ncfields)

    # SharedData -- that is the data structure that will be used by the
    # main thread to pass information down to the child thread(s)
    sdata = SharedData(
        name='sdata',
        npthreads=npthreads,
        cfields=cfields,
        ncfields=ncfields,
        pname=sregistry.make_name(prefix='tsdata')
    )
    sbase = sdata.indexed

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
    wrap = Conditional(CondEq(FieldFromPointer(sdata.symbolic_flag, sbase), 2), wrap)

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
            unpacks.append(DummyExpr(i, FieldFromPointer(i.base, sbase)))

    body = iet.body._rebuild(body=[wrap, Return(Null)], unpacks=unpacks)
    iet = ThreadCallable(iet.name, body, tparameter)

    return iet, {'includes': ['pthread.h']}


@iet_pass
def lower_async_calls(iet, track=None, sregistry=None):
    # Definitely there won't be AsyncCalls within ThreadCallables
    if isinstance(iet, ThreadCallable):
        return iet, {}

    # Create efuncs to initialize the SharedData objects
    efuncs = OrderedDict()
    for n in FindNodes(AsyncCall).visit(iet):
        if n.name in efuncs:
            continue

        assert n.name in track
        sdata = track[n.name]
        sbase = sdata.indexed
        name = sregistry.make_name(prefix='init_%s' % sdata.name)
        body = []
        for i in sdata.cfields:
            if i.is_AbstractFunction:
                body.append(
                    DummyExpr(FieldFromPointer(i._C_symbol, sbase), i._C_symbol)
                )
            else:
                body.append(DummyExpr(FieldFromPointer(i.base, sbase), i.base))
        body.extend([
            BlankLine,
            DummyExpr(FieldFromPointer(sdata.symbolic_flag, sbase), 1)
        ])
        parameters = sdata.cfields + (sdata,)
        efuncs[n.name] = Callable(name, body, 'void', parameters, 'static')

    # Transform AsyncCalls
    nqueues = 1  # Number of allocated asynchronous queues so far
    initialization = []
    finalization = []
    mapper = {}
    for n in FindNodes(AsyncCall).visit(iet):
        # Bind the abstract `sdata` to `n`
        name = sregistry.make_name(prefix='sdata')
        sdata = track[n.name]._rebuild(name=name)

        # The pthreads that will execute the AsyncCallable asynchronously
        name = sregistry.make_name(prefix='threads')
        threads = PThreadArray(name=name, npthreads=sdata.npthreads)

        # Call to `sdata` initialization Callable
        sbase = sdata.indexed
        d = threads.index
        arguments = []
        for a in n.arguments:
            if any(maybe_alias(a, i) for i in sdata.ncfields):
                continue
            elif isinstance(a, QueueID):
                # Different pthreads use different queues
                arguments.append(nqueues + d)
            else:
                arguments.append(a)
        # Each pthread has its own SharedData copy
        arguments.append(sbase + d)
        assert len(efuncs[n.name].parameters) == len(arguments)
        call0 = Call(efuncs[n.name].name, arguments)

        # Create pthreads
        tbase = threads.indexed
        call1 = Call('pthread_create', (
            tbase + d, Null, Call(n.name, [], is_indirect=True), sbase + d
        ))
        nqueues += threads.size

        # Glue together the initialization pieces
        if threads.size == 1:
            callback = lambda body: body
        else:
            callback = lambda body: Iteration(body, d, threads.size - 1)
        initialization.append(List(
            body=callback([call0, call1])
        ))

        # Finalization
        name = sregistry.make_name(prefix='shutdown')
        body = List(body=callback([
            While(CondEq(FieldFromComposite(sdata.symbolic_flag, sdata[d]), 2)),
            DummyExpr(FieldFromComposite(sdata.symbolic_flag, sdata[d]), 0),
            Call('pthread_join', (threads[d], Null))
        ]))
        efunc = efuncs[name] = make_callable(name, body)
        finalization.append(Call(name, efunc.parameters))

        # Activation
        if threads.size == 1:
            d = threads.index
            condition = CondNe(FieldFromComposite(sdata.symbolic_flag, sdata[d]), 1)
            activation = [While(condition)]
        else:
            d = Temp(name=sregistry.make_name(prefix=threads.index.name))
            condition = CondNe(FieldFromComposite(sdata.symbolic_flag, sdata[d]), 1)
            activation = [DummyExpr(d, 0),
                          While(condition, DummyExpr(d, (d + 1) % threads.size))]
        activation.extend([DummyExpr(FieldFromComposite(i.base, sdata[d]), i)
                           for i in sdata.ncfields])
        activation.append(
            DummyExpr(FieldFromComposite(sdata.symbolic_flag, sdata[d]), 2)
        )
        name = sregistry.make_name(prefix='activate')
        efunc = efuncs[name] = make_callable(name, activation)
        mapper[n] = Call(name, efunc.parameters)

    if mapper:
        # Inject activation
        iet = Transformer(mapper).visit(iet)

        # Inject initialization and finalization
        initialization = List(
            header=c.Comment("Fire up and initialize pthreads"),
            body=initialization + [BlankLine]
        )

        finalization = List(
            header=c.Comment("Wait for completion of pthreads"),
            body=finalization
        )

        body = iet.body._rebuild(
            body=[initialization] + list(iet.body.body) + [BlankLine, finalization]
        )
        iet = iet._rebuild(body=body)
    else:
        assert not initialization
        assert not finalization

    return iet, {'efuncs': tuple(efuncs.values())}


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

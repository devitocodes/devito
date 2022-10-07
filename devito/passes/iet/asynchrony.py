from collections import OrderedDict

import cgen as c

from devito.ir import (AsyncCall, AsyncCallable, BlankLine, Call, Callable,
                       Conditional, Dereference, DummyExpr, FindNodes, FindSymbols,
                       Iteration, List, PointerCast, Return, ThreadCallable,
                       Transformer, While)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import (CondEq, CondNe, FieldFromComposite, FieldFromPointer,
                              Null)
from devito.tools import DefaultOrderedDict, Bunch, split
from devito.types import Lock, Pointer, PThreadArray, QueueID, SharedData, Symbol

__all__ = ['pthreadify']


def pthreadify(graph, **kwargs):
    track = DefaultOrderedDict(lambda: Bunch(threads=None, sdata=None))

    lower_async_callables(graph, track=track, root=graph.root, **kwargs)
    lower_async_calls(graph, track=track, **kwargs)


@iet_pass
def lower_async_callables(iet, track=None, root=None, sregistry=None):
    if not isinstance(iet, AsyncCallable):
        return iet, {}

    n = len(track)

    # Determine the max number of threads that can run this `iet` in parallel
    locks = [i for i in iet.parameters if isinstance(i, Lock)]
    npthreads = min([i.size for i in locks], default=1)
    if npthreads > 1:
        npthreads = sregistry.make_npthreads(npthreads)

    # PthreadArray -- the symbol representing an array of pthreads, which will
    # execute the AsyncCallable asynchronously
    threads = track[iet.name].threads = PThreadArray(name='threads',
                                                     npthreads=npthreads)

    # The `cfields` are the constant fields, that is the fields whose value
    # definitely never changes across different executions of `ìet`; the
    # `ncfields` are instead the non-constant fields, that is the fields whose
    # value may or may not change across different calls to `iet`
    fields = iet.parameters
    defines = FindSymbols('defines').visit(root.body)
    ncfields, cfields = split(fields, lambda i: i in defines)

    # SharedData -- that is the data structure that will be used by the
    # main thread to pass information down to the child thread(s)
    sdata = track[iet.name].sdata = SharedData(name='sdata',
                                               npthreads=threads.size,
                                               cfields=cfields,
                                               ncfields=ncfields,
                                               pname='tsdata%d' % n)
    sbase = sdata.symbolic_base

    # Prepend the SharedData fields available upon thread activation
    preactions = [DummyExpr(i, FieldFromPointer(i.name, sbase)) for i in ncfields]
    preactions.append(BlankLine)

    # Append the flag reset
    postactions = [List(body=[
        BlankLine,
        DummyExpr(FieldFromPointer(sdata._field_flag, sbase), 1)
    ])]

    wrap = List(body=preactions + list(iet.body.body) + postactions)

    # The thread has work to do when it receives the signal that all locks have
    # been set to 0 by the main thread
    wrap = Conditional(CondEq(FieldFromPointer(sdata._field_flag, sbase), 2), wrap)

    # The thread keeps spinning until the alive flag is set to 0 by the main thread
    wrap = While(CondNe(FieldFromPointer(sdata._field_flag, sbase), 0), wrap)

    # pthread functions expect exactly one argument of type void*
    tparameter = Pointer(name='_%s' % sdata.name)

    # Unpack `sdata`
    unpacks = [PointerCast(sdata, tparameter), BlankLine]
    for i in cfields:
        if i.is_AbstractFunction:
            unpacks.append(Dereference(i, sdata))
        else:
            unpacks.append(DummyExpr(i, FieldFromPointer(i.name, sbase)))

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
        b = track[n.name]

        sdata = b.sdata
        sbase = sdata.symbolic_base
        name = sregistry.make_name(prefix='init_%s' % sdata.name)
        body = [DummyExpr(FieldFromPointer(i._C_name, sbase), i._C_symbol)
                for i in sdata.cfields]
        body.extend([BlankLine, DummyExpr(FieldFromPointer(sdata._field_flag, sbase), 1)])
        parameters = sdata.cfields + (sdata,)
        efuncs[n.name] = Callable(name, body, 'void', parameters, 'static')

    # Transform AsyncCalls
    nqueues = 1  # Number of allocated asynchronous queues so far
    initialization = []
    finalization = []
    mapper = {}
    for n in FindNodes(AsyncCall).visit(iet):
        # Create `sdata` and `threads` objects for `n`
        b = track[n.name]
        name = sregistry.make_name(prefix='sdata')
        sdata = b.sdata._rebuild(name=name)
        name = sregistry.make_name(prefix='threads')
        threads = b.threads._rebuild(name=name)

        # Call to `sdata` initialization Callable
        sbase = sdata.symbolic_base
        d = threads.index
        arguments = []
        for a in n.arguments:
            if a in sdata.ncfields:
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
        tbase = threads.symbolic_base
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
            header=c.Comment("Fire up and initialize `%s`" % threads.name),
            body=callback([call0, call1])
        ))

        # Finalization
        finalization.append(List(
            header=c.Comment("Wait for completion of `%s`" % threads.name),
            body=callback([
                While(CondEq(FieldFromComposite(sdata._field_flag, sdata[d]), 2)),
                DummyExpr(FieldFromComposite(sdata._field_flag, sdata[d]), 0),
                Call('pthread_join', (threads[d], Null))
            ])
        ))

        # Activation
        if threads.size == 1:
            d = threads.index
            condition = CondNe(FieldFromComposite(sdata._field_flag, sdata[d]), 1)
            activation = [While(condition)]
        else:
            d = Symbol(name=sregistry.make_name(prefix=threads.index.name))
            condition = CondNe(FieldFromComposite(sdata._field_flag, sdata[d]), 1)
            activation = [DummyExpr(d, 0),
                          While(condition, DummyExpr(d, (d + 1) % threads.size))]
        activation.extend([DummyExpr(FieldFromComposite(i.name, sdata[d]), i)
                           for i in sdata.ncfields])
        activation.append(DummyExpr(FieldFromComposite(sdata._field_flag, sdata[d]), 2))
        activation = List(
            header=[c.Line(), c.Comment("Activate `%s`" % threads.name)],
            body=activation,
            footer=c.Line()
        )
        mapper[n] = activation

    if mapper:
        # Inject activation
        iet = Transformer(mapper).visit(iet)

        # Inject initialization and finalization
        initialization.append(BlankLine)
        finalization.insert(0, BlankLine)
        body = iet.body._rebuild(body=initialization + list(iet.body.body) + finalization)
        iet = iet._rebuild(body=body)
    else:
        assert not initialization
        assert not finalization

    return iet, {'efuncs': tuple(efuncs.values())}

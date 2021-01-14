from collections import namedtuple

from cached_property import cached_property
from sympy import Or
import cgen as c

from devito.ir.iet.nodes import (BlankLine, Call, Callable, Conditional, Dereference,
                                 DummyExpr, Iteration, List, PointerCast, Return, While)
from devito.ir.iet.utils import derive_parameters, diff_parameters
from devito.symbolics import CondEq, CondNe, FieldFromComposite, FieldFromPointer, Macro
from devito.tools import as_tuple
from devito.types import PThreadArray, SharedData, Symbol, VoidPointer

__all__ = ['ElementalFunction', 'ElementalCall', 'make_efunc',
           'EntryFunction', 'ThreadFunction', 'make_thread_ctx']


# ElementalFunction machinery

class ElementalFunction(Callable):

    """
    A Callable performing a computation over an abstract convex iteration space.

    A Call to an ElementalFunction will "instantiate" such iteration space by
    supplying bounds and step increment for each Dimension listed in
    ``dynamic_parameters``.
    """

    is_ElementalFunction = True

    def __init__(self, name, body, retval, parameters=None, prefix=('static', 'inline'),
                 dynamic_parameters=None):
        super(ElementalFunction, self).__init__(name, body, retval, parameters, prefix)

        self._mapper = {}
        for i in as_tuple(dynamic_parameters):
            if i.is_Dimension:
                self._mapper[i] = (parameters.index(i.symbolic_min),
                                   parameters.index(i.symbolic_max))
            else:
                self._mapper[i] = (parameters.index(i),)

    @cached_property
    def dynamic_defaults(self):
        return {k: tuple(self.parameters[i] for i in v) for k, v in self._mapper.items()}

    def make_call(self, dynamic_args_mapper=None, incr=False, retobj=None,
                  is_indirect=False):
        return ElementalCall(self.name, list(self.parameters), dict(self._mapper),
                             dynamic_args_mapper, incr, retobj, is_indirect)


class ElementalCall(Call):

    def __init__(self, name, arguments=None, mapper=None, dynamic_args_mapper=None,
                 incr=False, retobj=None, is_indirect=False):
        self._mapper = mapper or {}

        arguments = list(as_tuple(arguments))
        dynamic_args_mapper = dynamic_args_mapper or {}
        for k, v in dynamic_args_mapper.items():
            tv = as_tuple(v)

            # Sanity check
            if k not in self._mapper:
                raise ValueError("`k` is not a dynamic parameter" % k)
            if len(self._mapper[k]) != len(tv):
                raise ValueError("Expected %d values for dynamic parameter `%s`, given %d"
                                 % (len(self._mapper[k]), k, len(tv)))
            # Create the argument list
            for i, j in zip(self._mapper[k], tv):
                arguments[i] = j if incr is False else (arguments[i] + j)

        super(ElementalCall, self).__init__(name, arguments, retobj, is_indirect)

    def _rebuild(self, *args, dynamic_args_mapper=None, incr=False,
                 retobj=None, **kwargs):
        # This guarantees that `ec._rebuild(arguments=ec.arguments) == ec`
        return super(ElementalCall, self)._rebuild(
            *args, dynamic_args_mapper=dynamic_args_mapper, incr=incr,
            retobj=retobj, **kwargs
        )

    @cached_property
    def dynamic_defaults(self):
        return {k: tuple(self.arguments[i] for i in v) for k, v in self._mapper.items()}


def make_efunc(name, iet, dynamic_parameters=None, retval='void', prefix='static'):
    """
    Shortcut to create an ElementalFunction.
    """
    return ElementalFunction(name, iet, retval, derive_parameters(iet), prefix,
                             dynamic_parameters)


# EntryFunction machinery

class EntryFunction(Callable):
    pass


# ThreadFunction machinery

ThreadCtx = namedtuple('ThreadCtx', 'threads sdata funcs init activate finalize')


class ThreadFunction(Callable):

    """
    A Callable executed asynchronously by a separate thread.
    """

    pass


def _make_threads(value, sregistry):
    name = sregistry.make_name(prefix='threads')

    if value is None:
        threads = PThreadArray(name=name, npthreads=1)
    else:
        npthreads = sregistry.make_npthreads(value)
        threads = PThreadArray(name=name, npthreads=npthreads)

    return threads


def _make_thread_init(threads, tfunc, isdata, sdata, sregistry):
    d = threads.index
    if threads.size == 1:
        callback = lambda body: body
    else:
        callback = lambda body: Iteration(body, d, threads.size - 1)

    # A unique identifier for each created pthread
    other_thread_pools = set(sregistry.npthreads) - {threads.size}
    pthreadid = 1 + sum(i.data for i in other_thread_pools) + d

    # Initialize `sdata`
    arguments = list(isdata.parameters)
    arguments[-2] = sdata.symbolic_base + d
    arguments[-1] = pthreadid
    call0 = Call(isdata.name, arguments)

    # Create pthreads
    call1 = Call('pthread_create', (threads.symbolic_base + d,
                                    Macro('NULL'),
                                    Call(tfunc.name, [], is_indirect=True),
                                    sdata.symbolic_base + d))

    threadsinit = List(
        header=c.Comment("Fire up and initialize `%s`" % threads.name),
        body=callback([call0, call1])
    )

    return threadsinit


def _make_thread_func(name, iet, root, threads, sregistry):
    # Create the SharedData, that is the data structure that will be used by the
    # main thread to pass information dows to the child thread(s)
    required, parameters, dynamic_parameters = diff_parameters(iet, root)
    parameters = sorted(parameters, key=lambda i: i.is_Function)  # Allow casting
    sdata = SharedData(name=sregistry.make_name(prefix='sdata'), npthreads=threads.size,
                       fields=required, dynamic_fields=dynamic_parameters)

    sbase = sdata.symbolic_base
    sid = sdata.symbolic_id

    # Create a Callable to initialize `sdata` with the known const values
    iname = 'init_%s' % sdata.dtype._type_.__name__
    ibody = [DummyExpr(FieldFromPointer(i._C_name, sbase), i._C_symbol)
             for i in parameters]
    ibody.extend([
        BlankLine,
        DummyExpr(FieldFromPointer(sdata._field_id, sbase), sid),
        DummyExpr(FieldFromPointer(sdata._field_flag, sbase), 1)
    ])
    iparameters = parameters + [sdata, sid]
    isdata = Callable(iname, ibody, 'void', iparameters, 'static')

    # Prepend the SharedData fields available upon thread activation
    preactions = [DummyExpr(i, FieldFromPointer(i.name, sbase))
                  for i in dynamic_parameters]

    # Append the flag reset
    postactions = [List(body=[
        BlankLine,
        DummyExpr(FieldFromPointer(sdata._field_flag, sbase), 1)
    ])]

    iet = List(body=preactions + [iet] + postactions)

    # The thread has work to do when it receives the signal that all locks have
    # been set to 0 by the main thread
    iet = Conditional(CondEq(FieldFromPointer(sdata._field_flag, sbase), 2), iet)

    # The thread keeps spinning until the alive flag is set to 0 by the main thread
    iet = While(CondNe(FieldFromPointer(sdata._field_flag, sbase), 0), iet)

    # pthread functions expect exactly one argument, a void*, and must return void*
    tretval = 'void*'
    tparameter = VoidPointer('_%s' % sdata.name)

    # Unpack `sdata`
    unpack = [PointerCast(sdata, tparameter), BlankLine]
    for i in parameters:
        if i.is_AbstractFunction:
            unpack.extend([Dereference(i, sdata), PointerCast(i)])
        else:
            unpack.append(DummyExpr(i, FieldFromPointer(i.name, sbase)))
    unpack.append(DummyExpr(sid, FieldFromPointer(sdata._field_id, sbase)))
    unpack.append(BlankLine)
    iet = List(body=unpack + [iet, BlankLine, Return(Macro('NULL'))])

    tfunc = ThreadFunction(name, iet, tretval, tparameter, 'static')

    return tfunc, isdata, sdata


def _make_thread_activate(threads, sdata, sync_ops, sregistry):
    if threads.size == 1:
        d = threads.index
    else:
        d = Symbol(name=sregistry.make_name(prefix=threads.index.name))

    sync_locks = [s for s in sync_ops if s.is_SyncLock]
    condition = Or(*([CondNe(s.handle, 2) for s in sync_locks] +
                     [CondNe(FieldFromComposite(sdata._field_flag, sdata[d]), 1)]))

    if threads.size == 1:
        activation = [While(condition)]
    else:
        activation = [DummyExpr(d, 0),
                      While(condition, DummyExpr(d, (d + 1) % threads.size))]

    activation.extend([DummyExpr(FieldFromComposite(i.name, sdata[d]), i)
                       for i in sdata.dynamic_fields])
    activation.extend([DummyExpr(s.handle, 0) for s in sync_locks])
    activation.append(DummyExpr(FieldFromComposite(sdata._field_flag, sdata[d]), 2))
    activation = List(
        header=[c.Line(), c.Comment("Activate `%s`" % threads.name)],
        body=activation,
        footer=c.Line()
    )

    return activation


def _make_thread_finalize(threads, sdata):
    d = threads.index
    if threads.size == 1:
        callback = lambda body: body
    else:
        callback = lambda body: Iteration(body, d, threads.size - 1)

    threadswait = List(
        header=c.Comment("Wait for completion of `%s`" % threads.name),
        body=callback([
            While(CondEq(FieldFromComposite(sdata._field_flag, sdata[d]), 2)),
            DummyExpr(FieldFromComposite(sdata._field_flag, sdata[d]), 0),
            Call('pthread_join', (threads[d], Macro('NULL')))
        ])
    )

    return threadswait


def make_thread_ctx(name, iet, root, npthreads, sync_ops, sregistry):
    """
    Shortcut to create a ThreadFunction and all support data structures and
    routines to implement communication between the main thread and the child
    threads executing the ThreadFunction.
    """
    threads = _make_threads(npthreads, sregistry)
    tfunc, isdata, sdata = _make_thread_func(name, iet, root, threads, sregistry)
    init = _make_thread_init(threads, tfunc, isdata, sdata, sregistry)
    activate = _make_thread_activate(threads, sdata, sync_ops, sregistry)
    finalize = _make_thread_finalize(threads, sdata)

    return ThreadCtx(threads, sdata, [tfunc, isdata], init, activate, finalize)

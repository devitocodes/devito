from collections import namedtuple

from cached_property import cached_property
from sympy import Or
import cgen as c

from devito.ir.iet.nodes import (BlankLine, Call, Callable, Conditional, DummyExpr,
                                 Iteration, List, While)
from devito.ir.iet.utils import derive_parameters
from devito.symbolics import (CondEq, CondNe, FieldFromComposite, FieldFromPointer,
                              Macro)
from devito.tools import as_tuple, split
from devito.types import PThreadArray, SharedData, Symbol

__all__ = ['ElementalFunction', 'ElementalCall', 'make_efunc',
           'ThreadFunction', 'make_thread_ctx']


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


ThreadCtx = namedtuple('ThreadCtx', 'threads sdata init tfunc activate finalize')


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


def _make_thread_init(threads, tfunc, sdata, sregistry):
    d = threads.index
    if threads.size == 1:
        callback = lambda body: body
    else:
        callback = lambda body: Iteration(body, d, threads.size - 1)

    base = list(sregistry.npthreads)
    base.remove(threads.size)
    idinit = DummyExpr(FieldFromComposite(sdata._field_id, sdata[d]),
                       1 + sum(i.data for i in base) + d)
    call = Call('pthread_create', (threads.symbolic_base + d,
                                   Macro('NULL'),
                                   Call(tfunc.name, [], is_indirect=True),
                                   sdata.symbolic_base + d))
    threadsinit = List(
        header=c.Comment("Fire up and initialize `%s`" % threads.name),
        body=callback([idinit, call])
    )

    return threadsinit


def _make_thread_func(name, iet, root, npthreads, sregistry):
    # Create the SharedData
    required = derive_parameters(iet)
    known = (root.parameters +
             tuple(i for i in required if i.is_Array and i._mem_shared))
    parameters, dynamic_parameters = split(required, lambda i: i in known)

    sdata = SharedData(name=sregistry.make_name(prefix='sdata'),
                       npthreads=npthreads, fields=dynamic_parameters)
    parameters.append(sdata)

    # Prepend the unwinded SharedData fields, available upon thread activation
    preactions = [DummyExpr(i, FieldFromPointer(i.name, sdata.symbolic_base))
                  for i in dynamic_parameters]
    preactions.append(DummyExpr(sdata.symbolic_id,
                                FieldFromPointer(sdata._field_id,
                                                 sdata.symbolic_base)))

    # Append the flag reset
    postactions = [List(body=[
        BlankLine,
        DummyExpr(FieldFromPointer(sdata._field_flag, sdata.symbolic_base), 1)
    ])]

    iet = List(body=preactions + [iet] + postactions)

    # Append the flag reset

    # The thread has work to do when it receives the signal that all locks have
    # been set to 0 by the main thread
    iet = Conditional(CondEq(FieldFromPointer(sdata._field_flag,
                                              sdata.symbolic_base), 2), iet)

    # The thread keeps spinning until the alive flag is set to 0 by the main thread
    iet = While(CondNe(FieldFromPointer(sdata._field_flag, sdata.symbolic_base), 0),
                iet)

    return ThreadFunction(name, iet, 'void', parameters, 'static'), sdata


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
                       for i in sdata.fields])
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
    tfunc, sdata = _make_thread_func(name, iet, root, npthreads, sregistry)
    init = _make_thread_init(threads, tfunc, sdata, sregistry)
    activate = _make_thread_activate(threads, sdata, sync_ops, sregistry)
    finalize = _make_thread_finalize(threads, sdata)

    return ThreadCtx(threads, sdata, init, tfunc, activate, finalize)

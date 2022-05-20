from collections import namedtuple

from cached_property import cached_property
from sympy import Or
import cgen as c
import numpy as np

from devito.ir.iet.nodes import (BlankLine, Call, Callable, Conditional, Dereference,
                                 DummyExpr, Iteration, List, PointerCast, Return, While,
                                 CallableBody)
from devito.ir.iet.utils import derive_parameters, diff_parameters
from devito.symbolics import CondEq, CondNe, FieldFromComposite, FieldFromPointer, Keyword
from devito.tools import as_tuple
from devito.types import Pointer, PThreadArray, SharedData, ThreadArray, Symbol

__all__ = ['ElementalFunction', 'ElementalCall', 'make_efunc',
           'EntryFunction', 'ThreadFunction', 'SharedDataInitFunction', 'make_thread_ctx',
           'DeviceFunction', 'DeviceCall']


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

    def __init__(self, name, body, sdata, **kwargs):
        sid = SharedData._symbolic_id
        sbase = sdata.indexed

        # If we got it from a reconstruction, we need to remove the `unpacks` first,
        # as we're going to reconstruct the whole list from scratch
        body = body._rebuild(unpacks=None)

        # The pthread entry point expects exactly one argument -- of type void* --
        # and must return a void*
        retval = 'void*'
        parameter = Pointer(name='_%s' % sdata.name, dtype=np.void)

        # Unpack `sdata`'s known fields
        unpack = [
            PointerCast(sdata, parameter),
            BlankLine,
            DummyExpr(sid, FieldFromPointer(sdata._field_id, sbase), init=True),
        ]

        # Derive all "static parameters", that is the parameters that must be
        # passed to the pthread, via `sdata`, from the caller
        parameters = derive_parameters((unpack, body))

        #TODO: DROP???
        parameters = sorted(parameters, key=lambda i: i.is_Function)

        # A struct for the static fields
        idata = ThreadArray(
            name='%s_%s' % (sdata.name, sdata._field_constant),
            npthreads=sdata.size,
            fields=parameters
        )

        # Unpack `sdata`'s static fields
        unpack.extend([
            PointerCast(idata, FieldFromPointer(sdata._field_constant, sbase)),
            BlankLine
        ])
        for i in parameters:
            if i.is_AbstractFunction:
                unpack.append(Dereference(i, idata))
            else:
                unpack.append(
                    DummyExpr(i, FieldFromPointer(i.name, idata.indexed), init=True)
                )

        body = body._rebuild(unpacks=unpack)

        super().__init__(name, body, retval, parameter, 'static')

        self.ssparameters = parameters
        self.sdata = sdata
        self.idata = idata

    @cached_property
    def ifunc(self):
        return SharedDataInitFunction(self)


class SharedDataInitFunction(Callable):

    """
    A callable to initialize the SharedData object created by an Operator
    and plumbed down to a ThreadFunction.
    """

    def __init__(self, tfunc, **kwargs):
        sid = SharedData._symbolic_id

        sdata = tfunc.sdata
        idata = tfunc.idata
        parameters = tfunc.ssparameters

        sbase = sdata.indexed
        ibase = idata.indexed

        name = 'init_%s' % sdata.dtype._type_.__name__

        iet = [DummyExpr(FieldFromPointer(i._C_name, ibase), i._C_symbol)
               for i in parameters]
        iet.extend([
            BlankLine,
            DummyExpr(FieldFromPointer(sdata._field_constant, sbase), ibase),
            DummyExpr(FieldFromPointer(sdata._field_id, sbase), sid),
            DummyExpr(FieldFromPointer(sdata._field_flag, sbase), 1)
        ])

        parameters = parameters + [sdata, idata, sid]

        super().__init__(name, iet, 'void', parameters, 'static')

        self.caller = tfunc.name


def _make_threads(value, sregistry):
    name = sregistry.make_name(prefix='threads')

    base_id = 1 + sum(i.size for i in sregistry.npthreads)

    if value is None:
        # The npthreads Symbol isn't actually used, but we record the fact
        # that some new pthreads have been allocated
        sregistry.make_npthreads(1)
        npthreads = 1
    else:
        npthreads = sregistry.make_npthreads(value)

    threads = PThreadArray(name=name, npthreads=npthreads, base_id=base_id)

    return threads


def _make_thread_init(threads, tfunc, ifunc, sregistry):
    d = threads.index
    if threads.size == 1:
        callback = lambda body: body
    else:
        callback = lambda body: Iteration(body, d, threads.size - 1)

    # A unique identifier for each created pthread
    pthreadid = d + threads.base_id

    # Initialize `sdata`
    arguments = list(ifunc.parameters)
    arguments[-3] = tfunc.sdata.indexed + d
    arguments[-2] = tfunc.idata.indexed + d
    arguments[-1] = pthreadid
    call0 = Call(ifunc.name, arguments)

    # Create pthreads
    call1 = Call('pthread_create', (threads.indexed + d,
                                    Keyword('NULL'),
                                    Call(tfunc.name, [], is_indirect=True),
                                    tfunc.sdata.indexed + d))

    threadsinit = List(
        header=c.Comment("Fire up and initialize `%s`" % threads.name),
        body=callback([call0, call1])
    )

    return threadsinit


def _make_thread_func(name, iet, root, threads, sregistry):
    sid = SharedData._symbolic_id

    # Create the SharedData, that is the data structure that will be used by the
    # main thread to pass information down to the child thread(s)
    dynamic_parameters = diff_parameters(iet, root, [sid])
    sdata = SharedData(
        name=sregistry.make_name(prefix='sdata'),
        npthreads=threads.size,
        dynamic_fields=dynamic_parameters
    )
    sbase = sdata.indexed

    # Prepend the `sdata` dynamic fields
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

    # Finally, wrap everything within a ThreadFunction
    iet = CallableBody([iet, Return(Keyword('NULL'))])
    tfunc = ThreadFunction(name, iet, sdata)

    # Create a Callable to initialize `sdata` with the static fields
    ifunc = tfunc.ifunc

    return tfunc, ifunc, sdata


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
            Call('pthread_join', (threads[d], Keyword('NULL')))
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
    tfunc, ifunc, sdata = _make_thread_func(name, iet, root, threads, sregistry)
    init = _make_thread_init(threads, tfunc, ifunc, sregistry)
    activate = _make_thread_activate(threads, sdata, sync_ops, sregistry)
    finalize = _make_thread_finalize(threads, sdata)

    return ThreadCtx(threads, sdata, [tfunc, ifunc], init, activate, finalize)


# DeviceFunction machinery


class DeviceFunction(Callable):

    """
    A Callable executed asynchronously on a device.
    """

    def __init__(self, name, body, retval='void', parameters=None, prefix='__global__'):
        super().__init__(name, body, retval, parameters=parameters, prefix=prefix)


class DeviceCall(Call):

    """
    A call to an external function executed asynchronously on a device.
    """

    pass

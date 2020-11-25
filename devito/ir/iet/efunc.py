from cached_property import cached_property

from devito.ir.iet.nodes import (BlankLine, Call, Callable, Conditional, DummyExpr,
                                 List, While)
from devito.ir.iet.utils import derive_parameters
from devito.symbolics import CondEq, CondNe, FieldFromPointer
from devito.tools import as_tuple, split
from devito.types import SharedData

__all__ = ['ElementalFunction', 'ElementalCall', 'make_efunc',
           'ThreadFunction', 'make_tfunc']


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


class ThreadFunction(Callable):

    """
    A Callable executed asynchronously by a separate thread.
    """

    def __init__(self, name, body, retval, parameters=None, prefix=('static',),
                 sdata=None):
        super().__init__(name, body, retval, parameters, prefix)
        self.sdata = sdata


def make_tfunc(name, iet, root, threads, sregistry):
    """
    Shortcut to create a ThreadFunction and all support data structures and
    routines to implement communication between the main thread and the child
    threads executing the ThreadFunction.
    """
    #TODO: threads -> npthreads

    # Create the SharedData
    required = derive_parameters(iet)
    known = (root.parameters +
             tuple(i for i in required if i.is_Array and i._mem_shared))
    parameters, dynamic_parameters = split(required, lambda i: i in known)

    sdata = SharedData(name=sregistry.make_name(prefix='sdata'),
                       npthreads=threads.size, fields=dynamic_parameters)
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

    return ThreadFunction(name, iet, 'void', parameters, 'static', sdata)

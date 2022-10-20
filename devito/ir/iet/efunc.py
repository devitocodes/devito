from cached_property import cached_property

from devito.ir.iet.nodes import Call, Callable
from devito.ir.iet.utils import derive_parameters
from devito.symbolics import uxreplace
from devito.tools import as_tuple

__all__ = ['ElementalFunction', 'ElementalCall', 'make_efunc', 'make_callable',
           'EntryFunction', 'AsyncCallable', 'AsyncCall', 'ThreadCallable',
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


# Callable machinery


def make_callable(name, iet, retval='void', prefix='static'):
    """
    Utility function to create a Callable from an IET.
    """
    parameters = derive_parameters(iet)
    return Callable(name, iet, retval, parameters=parameters, prefix=prefix)


# EntryFunction machinery

class EntryFunction(Callable):
    pass


# AsyncCallables machinery

class AsyncCallable(Callable):

    """
    A Callable that is meant to be executed asynchronously by a thread.
    """

    def __init__(self, name, body, parameters=None, prefix='static'):
        super().__init__(name, body, 'void', parameters=parameters, prefix=prefix)


class AsyncCall(Call):
    pass


class ThreadCallable(Callable):

    """
    A Callable executed asynchronously by a thread.
    """

    def __init__(self, name, body, parameters=None, prefix='static'):
        super().__init__(name, body, 'void*', parameters=parameters, prefix=prefix)


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

    def __init__(self, name, arguments=None, **kwargs):
        # Explicitly convert host pointers into device pointers
        processed = []
        for a in arguments:
            try:
                f = a.function
            except AttributeError:
                processed.append(a)
                continue
            if f._mem_mapped:
                processed.append(uxreplace(a, {f.indexed: f.dmap}))
            else:
                processed.append(a)

        super().__init__(name, arguments=processed, **kwargs)

from functools import cached_property

from devito.ir.iet.nodes import Call, Callable
from devito.ir.iet.utils import derive_parameters
from devito.symbolics import uxreplace
from devito.tools import as_tuple

__all__ = ['ElementalFunction', 'ElementalCall', 'make_efunc', 'make_callable',
           'EntryFunction', 'AsyncCallable', 'AsyncCall', 'ThreadCallable',
           'DeviceFunction', 'DeviceCall', 'KernelLaunch', 'CommCallable']


# ElementalFunction machinery

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

        super().__init__(name, arguments, retobj, is_indirect)

    def _rebuild(self, *args, dynamic_args_mapper=None, incr=False,
                 retobj=None, **kwargs):
        # This guarantees that `ec._rebuild(arguments=ec.arguments) == ec`
        return super()._rebuild(
            *args, dynamic_args_mapper=dynamic_args_mapper, incr=incr,
            retobj=retobj, **kwargs
        )

    @cached_property
    def dynamic_defaults(self):
        return {k: tuple(self.arguments[i] for i in v) for k, v in self._mapper.items()}


class ElementalFunction(Callable):

    """
    A Callable performing a computation over an abstract convex iteration space.

    A Call to an ElementalFunction will "instantiate" such iteration space by
    supplying bounds and step increment for each Dimension listed in
    ``dynamic_parameters``.
    """
    _Call_cls = ElementalCall

    is_ElementalFunction = True

    def __init__(self, name, body, retval='void', parameters=None, prefix=('static',),
                 dynamic_parameters=None):
        super().__init__(name, body, retval, parameters, prefix)

        self._mapper = {}
        for i in as_tuple(dynamic_parameters):
            if i.is_Dimension:
                self._mapper[i] = (parameters.index(i.symbolic_min),
                                   parameters.index(i.symbolic_max))
            else:
                self._mapper[i] = (parameters.index(i),)

    @classmethod
    def make(cls, name, body):
        parameters = derive_parameters(body)
        return cls(name, body, parameters=parameters)

    @cached_property
    def dynamic_defaults(self):
        return {k: tuple(self.parameters[i] for i in v) for k, v in self._mapper.items()}

    def make_call(self, dynamic_args_mapper=None, incr=False, retobj=None,
                  is_indirect=False):
        return self._Call_cls(self.name, list(self.parameters), dict(self._mapper),
                              dynamic_args_mapper, incr, retobj, is_indirect)


def make_efunc(name, iet, dynamic_parameters=None, retval='void', prefix='static',
               efunc_type=ElementalFunction):
    """
    Shortcut to create an ElementalFunction.
    """
    return efunc_type(name, iet, retval=retval,
                      parameters=derive_parameters(iet), prefix=prefix,
                      dynamic_parameters=dynamic_parameters)


# Callable machinery


def make_callable(name, iet, retval='void', prefix='static'):
    """
    Utility function to create a Callable from an IET.
    """
    parameters = derive_parameters(iet, ordering='canonical')
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

    def __init__(self, name, body, parameters):
        super().__init__(name, body, 'void*', parameters=parameters, prefix='static')

        # Sanity checks
        # By construction, the first unpack statement of a ThreadCallable must
        # be the PointerCast that makes `sdata` available in the local scope
        assert len(body.unpacks) > 0
        v = body.unpacks[0]
        assert v.is_PointerCast
        self.sdata = v.function


# DeviceFunction machinery


class DeviceFunction(Callable):

    """
    A Callable executed asynchronously on a device.
    """

    def __init__(self, name, body, retval='void', parameters=None,
                 prefix='__global__', templates=None, attributes=None):
        super().__init__(name, body, retval, parameters=parameters, prefix=prefix,
                         templates=templates, attributes=attributes)


class DeviceCall(Call):

    """
    A call to a function executed asynchronously on a device.
    """

    def __init__(self, name, arguments=None, **kwargs):
        # Explicitly convert host pointers into device pointers
        processed = []
        for a in as_tuple(arguments):
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


class KernelLaunch(DeviceCall):

    """
    A call to an asynchronous device kernel.
    """

    def __init__(self, name, grid, block, shm=0, stream=None,
                 arguments=None, writes=None, templates=None):
        super().__init__(name, arguments=arguments, writes=writes,
                         templates=templates)

        # Kernel launch arguments
        self.grid = grid
        self.block = block
        self.shm = shm
        self.stream = stream

    def __repr__(self):
        return 'Launch[%s]<<<(%s)>>>' % (self.name,
                                         ','.join(str(i.name) for i in self.writes))

    @cached_property
    def functions(self):
        launch_args = (self.grid, self.block,)
        if self.stream is not None:
            launch_args += (self.stream.function,)
        return super().functions + launch_args

    @cached_property
    def expr_symbols(self):
        launch_symbols = (self.grid, self.block)
        if self.stream is not None:
            launch_symbols += (self.stream,)
        return super().expr_symbols + launch_symbols


# Other relevant Callable subclasses

class CommCallable(Callable):
    pass

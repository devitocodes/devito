from dataclasses import dataclass
from functools import cached_property
from itertools import chain

from devito.ir.iet.nodes import Call, Callable, CallableBody, DummyExpr
from devito.ir.iet.utils import derive_parameters
from devito.symbolics import FieldFromPointer, uxreplace
from devito.tools import as_tuple
from devito.types import Region

__all__ = [
    'AsyncCall',
    'AsyncCallable',
    'CommCallable',
    'DeviceCall',
    'DeviceFunction',
    'EFuncMeta',
    'ElementalCall',
    'ElementalFunction',
    'EntryFunction',
    'KernelLaunch',
    'ThreadCallable',
    'make_callable',
    'make_efunc',
]


@dataclass(frozen=True)
class EFuncMeta:

    body: object = None
    efuncs: tuple = ()
    includes: tuple = ()
    namespaces: tuple = ()
    libs: tuple = ()

    @classmethod
    def compose(cls, *items):
        items = tuple(items)

        if not items:
            return cls()

        return cls(
            body=items[-1].body,
            efuncs=tuple(chain.from_iterable(i.efuncs for i in items)),
            includes=tuple(chain.from_iterable(i.includes for i in items)),
            namespaces=tuple(chain.from_iterable(i.namespaces for i in items)),
            libs=tuple(chain.from_iterable(i.libs for i in items))
        )

    def __iter__(self):
        yield self.body
        yield self.efuncs
        yield self.includes
        yield self.namespaces
        yield self.libs


# ElementalFunction machinery

class ElementalCall(Call):

    @cached_property
    def region(self):
        # The argument may select a record, e.g. `&regions[i]`
        for a in self.arguments:
            try:
                if isinstance(a.function, Region):
                    return a
            except AttributeError:
                pass

        return None


class ElementalFunction(Callable):

    """
    A Callable performing a computation over an abstract convex iteration space.

    A Region parameter, when present, carries the bounds and scalar controls
    used to instantiate the iteration space at each Call.
    """
    _Call_cls = ElementalCall

    is_ElementalFunction = True

    def __init__(self, name, body, retval='void', parameters=None, prefix=('static',)):
        super().__init__(name, body, retval, parameters, prefix)

    @classmethod
    def make(cls, name, body):
        parameters = derive_parameters(body)
        return cls(name, body, parameters=parameters)

    @cached_property
    def region(self):
        return next((i for i in self.parameters if isinstance(i, Region)), None)

    def make_call(self, region=None, retobj=None, is_indirect=False):
        if region is not None and self.region is None:
            raise ValueError("No region parameter")

        arguments = [region if region is not None and i is self.region else i
                     for i in self.parameters]

        return self._Call_cls(self.name, arguments, retobj=retobj,
                              is_indirect=is_indirect)


def make_efunc(name, iet, region=None, retval='void', prefix='static',
               efunc_type=ElementalFunction):
    """
    Shortcut to create an ElementalFunction, optionally unpacking a Region.

    E.g., `compute(..., x_m, x_M)` becomes `compute(..., region)`, with local
    definitions `x_m = region->x_m` and `x_M = region->x_M`.
    """
    if region is not None:
        if not isinstance(iet, CallableBody):
            iet = CallableBody(iet)

        # Unpack all declared fields so subsequent lowering can use them
        unpacks = tuple(DummyExpr(i, FieldFromPointer(i.name, region), init=True)
                        for i in region.symbols)
        iet = iet._rebuild(unpacks=iet.unpacks + unpacks)

    return efunc_type(name, iet, retval=retval,
                      parameters=derive_parameters(iet), prefix=prefix)


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
        return 'Launch[{}]<<<({})>>>'.format(
            self.name,
            ','.join(str(i.name) for i in self.writes)
        )

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

from ctypes import c_int, c_double, c_void_p

from devito.types import CompositeObject, LocalObject, Indexed, Symbol
from devito.types.basic import IndexedData
from devito.tools import Pickable

__all__ = ['Timer', 'VoidPointer', 'VolatileInt', 'c_volatile_int',
           'c_volatile_int_p', 'FIndexed', 'Wildcard', 'Global']


class Timer(CompositeObject):

    def __init__(self, name, sections):
        super().__init__(name, 'profiler', [(i, c_double) for i in sections])

    def reset(self):
        for i in self.fields:
            setattr(self.value._obj, i, 0.0)
        return self.value

    @property
    def total(self):
        return sum(getattr(self.value._obj, i) for i in self.fields)

    @property
    def sections(self):
        return self.fields

    def _arg_values(self, **kwargs):
        values = super()._arg_values(**kwargs)

        # Reset timer
        for i in self.fields:
            setattr(values[self.name]._obj, i, 0.0)

        return values

    # Pickling support
    _pickle_args = ['name', 'sections']


class VoidPointer(LocalObject):

    dtype = type('void*', (c_void_p,), {})

    def __init__(self, name):
        self.name = name

    # Pickling support
    _pickle_args = ['name']


class VolatileInt(Symbol):

    @property
    def _C_typedata(self):
        return 'volatile int'

    _C_typename = _C_typedata

    @property
    def _C_ctype(self):
        return c_volatile_int


class Wildcard(Symbol):

    """
    A special Symbol used by the compiler to generate ad-hoc code
    (e.g. to work around known bugs in jit-compilers).
    """

    pass


class FIndexed(Indexed, Pickable):

    """
    A flatten Indexed with functional (primary) and indexed (secondary) representations.

    Examples
    --------
    Consider the Indexed `u[x, y]`. The corresponding FIndexed's functional representation
    is `u(x, y)`. This is a multidimensional representation, just like any other Indexed.
    The corresponding indexed (secondary) represenation is instead flatten, that is
    `uX[x*ny + y]`, where `X` is a string provided by the caller.
    """

    def __new__(cls, indexed, pname):
        plabel = Symbol(name=pname, dtype=indexed.dtype)
        base = IndexedData(plabel, shape=indexed.shape, function=indexed.function)
        obj = super().__new__(cls, base, *indexed.indices)

        obj.indexed = indexed
        obj.pname = pname

        return obj

    def __repr__(self):
        return "%s(%s)" % (self.name, ", ".join(str(i) for i in self.indices))

    __str__ = __repr__

    @property
    def name(self):
        return self.function.name

    # Pickling support
    _pickle_args = ['indexed', 'pname']
    __reduce_ex__ = Pickable.__reduce_ex__


class Global(Symbol):

    """
    A special Symbol representing global variables.
    """

    pass


# ctypes subtypes

class c_volatile_int(c_int):
    pass


class c_volatile_int_p(c_void_p):
    pass

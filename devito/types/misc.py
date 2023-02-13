from ctypes import c_double, c_void_p

import numpy as np
from sympy.core.core import ordering_of_classes

from devito.types import CompositeObject, Indexed, Symbol
from devito.types.basic import IndexedData
from devito.tools import Pickable, as_tuple

__all__ = ['Timer', 'Pointer', 'VolatileInt', 'FIndexed', 'Wildcard',
           'Global', 'Hyperplane', 'Indirection', 'Temp', 'Jump']


class Timer(CompositeObject):

    __rargs__ = ('name', 'sections')

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


class VolatileInt(Symbol):
    is_volatile = True


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

    __rargs__ = ('indexed', 'pname')
    __rkwargs__ = ('strides',)

    def __new__(cls, indexed, pname, strides=None):
        plabel = Symbol(name=pname, dtype=indexed.dtype)
        base = IndexedData(plabel, None, function=indexed.function)
        obj = super().__new__(cls, base, *indexed.indices)

        obj.indexed = indexed
        obj.pname = pname
        obj.strides = as_tuple(strides)

        return obj

    def __repr__(self):
        return "%s(%s)" % (self.name, ", ".join(str(i) for i in self.indices))

    __str__ = __repr__

    def _hashable_content(self):
        return super()._hashable_content() + (self.strides,)

    @property
    def name(self):
        return self.function.name

    @property
    def free_symbols(self):
        # The functional representation of the FIndexed "hides" the strides, which
        # are however actual free symbols of the object, since they contribute to
        # the address calculation just like all other free_symbols
        return (super().free_symbols |
                set().union(*[i.free_symbols for i in self.strides]))

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class Global(Symbol):

    """
    A special Symbol representing global variables.
    """

    pass


class Hyperplane(tuple):

    """
    A collection of Dimensions defining an hyperplane.
    """

    @property
    def _defines(self):
        return frozenset().union(*[i._defines for i in self])


class Pointer(Symbol):

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('dtype', c_void_p)

    @property
    def _C_ctype(self):
        # `dtype` is a ctypes-derived type!
        return self.dtype


class Indirection(Symbol):

    """
    An Indirection is a Symbol that holds a value used to indirectly access
    an Indexed.

    Examples
    --------
    Below an Indirection, `ofs`, used to access an array `a`.

        ofs = offsets[time + 1]
        v = a[ofs]
    """

    __rkwargs__ = Symbol.__rkwargs__ + ('mapped',)

    def __new__(cls, name=None, mapped=None, dtype=np.uint64, is_const=True):
        obj = super().__new__(cls, name=name, dtype=dtype, is_const=is_const)
        obj.mapped = mapped

        return obj


class Temp(Symbol):

    """
    A Temp is a Symbol used by compiler passes to store locally-constructed
    temporary expressions.
    """

    # Just make sure the SymPy args ordering is the same regardless of whether
    # the arguments are Symbols or Temps
    ordering_of_classes.insert(ordering_of_classes.index('Symbol') + 1, 'Temp')


class Jump(object):

    """
    Mixin class for symbolic objects representing jumps in the control flow,
    such as return and break statements.
    """

    pass

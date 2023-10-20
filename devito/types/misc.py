from ctypes import c_double, c_void_p

import numpy as np
import sympy
from sympy.core.core import ordering_of_classes

from devito.types import Array, CompositeObject, Indexed, Symbol
from devito.types.basic import IndexedData
from devito.tools import Pickable, as_tuple

__all__ = ['Timer', 'Pointer', 'VolatileInt', 'FIndexed', 'Wildcard', 'Fence',
           'Global', 'Hyperplane', 'Indirection', 'Temp', 'TempArray', 'Jump',
           'nop', 'WeakFence', 'CriticalRegion']


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

    __rargs__ = ('base', '*indices')
    __rkwargs__ = ('strides',)

    def __new__(cls, base, *args, strides=None):
        obj = super().__new__(cls, base, *args)
        obj.strides = as_tuple(strides)

        return obj

    @classmethod
    def from_indexed(cls, indexed, pname, strides=None):
        label = Symbol(name=pname, dtype=indexed.dtype)
        base = IndexedData(label, None, function=indexed.function)
        return FIndexed(base, *indexed.indices, strides=strides)

    def __repr__(self):
        return "%s(%s)" % (self.name, ", ".join(str(i) for i in self.indices))

    __str__ = __repr__

    def _hashable_content(self):
        return super()._hashable_content() + (self.strides,)

    func = Pickable._rebuild

    @property
    def name(self):
        return self.function.name

    @property
    def pname(self):
        return self.base.name

    @property
    def free_symbols(self):
        # The functional representation of the FIndexed "hides" the strides, which
        # are however actual free symbols of the object, since they contribute to
        # the address calculation just like all other free_symbols
        return (super().free_symbols |
                set().union(*[i.free_symbols for i in self.strides]))

    func = Pickable._rebuild

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

    def __new__(cls, name=None, mapped=None, dtype=np.uint64, is_const=True,
                **kwargs):
        obj = super().__new__(cls, name=name, dtype=dtype, is_const=is_const,
                              **kwargs)
        obj.mapped = mapped

        return obj


class Temp(Symbol):

    """
    A Temp is a Symbol used by compiler passes to store intermediate
    sub-expressions.
    """

    # Just make sure the SymPy args ordering is the same regardless of whether
    # the arguments are Symbols or Temps
    ordering_of_classes.insert(ordering_of_classes.index('Symbol') + 1, 'Temp')


class TempArray(Array):

    """
    A TempArray is an Array used by compiler passes to store intermediate
    sub-expressions.
    """

    pass


class Fence(object):

    """
    Mixin class for generic "fence" objects.

    A Fence is an object that enforces an ordering constraint on the
    surrounding operations: the operations issued before the Fence are
    guaranteed to be scheduled before operations issued after the Fence.

    The meaning of "operation" and its relationship with the concept of
    termination depends on the actual Fence subclass.

    For example, operations could be Eq's. A Fence will definitely impair
    topological sorting such that, e.g.

        Eq(A)
        Fence
        Eq(B)

    *cannot* get transformed into

        Eq(A)
        Eq(B)
        Fence

    However, a simple Fence won't dictate whether or not Eq(A) should also
    terminate before Eq(B).
    """

    pass


class Jump(Fence):

    """
    Mixin class for symbolic objects representing jumps in the control flow,
    such as return and break statements.
    """

    pass


class WeakFence(sympy.Function, Fence):

    """
    The weakest of all possible fences.

    Equations cannot be moved across a WeakFence.
    However an operation initiated before a WeakFence can terminate at any
    point in time.
    """

    pass


class CriticalRegion(sympy.Function, Fence):

    """
    A fence that either opens or closes a "critical sequence of Equations".

    There always are two CriticalRegions for each critical sequence of Equations:

        * `CriticalRegion(init)`: opens the critical sequence
        * `CriticalRegion(end)`: closes the critical sequence

    `CriticalRegion(end)` must follow `CriticalRegion(init)`.

    A CriticalRegion implements a strong form of fencing:

        * Equations within a critical sequence cannot be moved outside of
          the opening and closing CriticalRegions.
            * However, internal rearrangements are possible
        * An asynchronous operation initiated within the critial sequence must
          terminate before re-entering the opening CriticalRegion.
    """

    def __init__(self, opening, **kwargs):
        opening = bool(opening)

        sympy.Function.__init__(opening)
        self.opening = opening

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           'OPEN' if self.opening else 'CLOSE')

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    @property
    def closing(self):
        return not self.opening


nop = sympy.Function('NOP')
"""
A wildcard for use in the RHS of Eqs that encode some kind of semantics
(e.g., a synchronization operation) but no computation.
"""

from ctypes import c_double, c_void_p

import numpy as np
import sympy
try:
    from sympy.core.core import ordering_of_classes
except ImportError:
    # Moved in 1.13
    from sympy.core.basic import ordering_of_classes

from devito.types import Array, CompositeObject, Indexed, Symbol, LocalObject
from devito.types.basic import IndexedData
from devito.tools import CustomDtype, Pickable, frozendict

__all__ = ['Timer', 'Pointer', 'VolatileInt', 'FIndexed', 'Wildcard', 'Fence',
           'Global', 'Hyperplane', 'Indirection', 'Temp', 'TempArray', 'Jump',
           'nop', 'WeakFence', 'CriticalRegion', 'Auto', 'AutoRef', 'auto',
           'size_t']


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
    An FIndexed is a symbolic object used to represent a multidimensional
    array in symbolic equations. It is a subclass of Indexed, and as such
    it has a base (the symbol representing the array) and a number of indices.

    However, unlike Indexed, the representation of an FIndexed is functional,
    e.g., `u(x, y)`, rather than explicit, e.g., `u[x, y]`.

    An FIndexed also carries the necessary information to generate a 1-dimensional
    representation of the array, which is necessary when dealing with actual
    memory accesses. For example, an FIndexed carries the strides of the array,
    which ultimately allow to compute the actual memory address of an element.
    For example, the FIndexed `u(x, y)` corresponds to the indexed representation
    `u[x*ny + y]`, where `ny` is the stride of the array `u` along the y-axis.
    """

    __rargs__ = ('base', '*indices')
    __rkwargs__ = ('strides_map', 'accessor')

    def __new__(cls, base, *args, strides_map=None, accessor=None):
        obj = super().__new__(cls, base, *args)
        obj.strides_map = frozendict(strides_map or {})
        obj.accessor = accessor

        return obj

    def __repr__(self):
        return "%s(%s)" % (self.name, ", ".join(str(i) for i in self.indices))

    __str__ = __repr__

    def _hashable_content(self):
        accessor = self.accessor or 0  # Avoids TypeError inside sympy.Basic.compare
        return super()._hashable_content() + (self.strides, accessor)

    func = Pickable._rebuild

    @property
    def name(self):
        return self.base.name

    @property
    def strides(self):
        return tuple(self.strides_map.values())

    @property
    def free_symbols(self):
        # The functional representation of the FIndexed "hides" the strides, which
        # are however actual free symbols of the object, since they contribute to
        # the address calculation just like all other free_symbols
        return (super().free_symbols |
                set().union(*[i.free_symbols for i in self.strides]))

    def bind(self, pname):
        """
        Generate a 2-tuple:

            * A macro which expands to the 1-dimensional representation of the
              FIndexed, e.g. `aL0(t,x,y) -> a[(t)*x_stride0 + (x)*y_stride0 + (y)]`
            * A new FIndexed, with the same indices as `self`, but with a new
              base symbol named after `pname`, e.g. `aL0(t, x+1, y-2)`, where
              `aL0` is given by the `pname`.
        """
        b = self.base
        f = self.function
        strides_map = self.strides_map

        # TODO: resolve circular import. This is a tough one though, as it
        # requires a complete rethinking of `symbolics` vs `types` folders
        from devito.symbolics import DefFunction, MacroArgument

        macroargnames = [d.name for d in f.dimensions]
        macroargs = [MacroArgument(i) for i in macroargnames]

        items = [m*strides_map[d] for m, d in zip(macroargs, f.dimensions[1:])]
        items.append(MacroArgument(f.dimensions[-1].name))

        define = DefFunction(pname, macroargnames)
        expr = Indexed(b, sympy.Add(*items, evaluate=False))

        label = Symbol(name=pname, dtype=self.dtype)
        accessor = IndexedData(label, None, function=f)
        findexed = self.func(accessor=accessor)

        return ((define, expr), findexed)

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


class Pointer(LocalObject):

    __rkwargs__ = LocalObject.__rkwargs__ + ('dtype',)

    def __init__(self, *args, dtype=c_void_p, **kwargs):
        super().__init__(*args, **kwargs)
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype


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

    is_autopaddable = True

    def __padding_setup__(self, **kwargs):
        padding = kwargs.pop('padding', None)
        if padding is None:
            padding = self.__padding_setup_smart__(**kwargs)
        return super().__padding_setup__(padding=padding, **kwargs)


class Fence:

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


# *** C/CXX support types

size_t = CustomDtype('size_t')

# NOTE: In C++, `auto` is a type specifier more than a type itself, but
# it's a distinction we can afford to ignore, at least for now
auto = CustomDtype('auto')
auto_ref = CustomDtype('auto', modifier='&')


class Auto(LocalObject):
    dtype = auto


class AutoRef(LocalObject, sympy.Expr):
    dtype = auto_ref

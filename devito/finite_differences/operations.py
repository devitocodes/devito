import sympy
import numpy as np

import devito
from devito.symbolics.search import retrieve_functions
from devito.symbolics.extended_sympy import FrozenExpr

__all__ = ['Mul', 'Add', 'Pow']


def fd_parameters(obj):
    """
    Process a set of Mul, Add, Pow expression and outputs
    the space_order, time_order of the whole expression.
    As the leading order is always the smallest order, if multiple
    orders are found the lowest is chosen
    """
    func = list(retrieve_functions(obj))
    time_dims = ()
    sp_dims = ()
    so = 100
    to = 100
    is_time = False
    stagg = (0, 0, 0, 0)
    dtype = None
    # Filter expressions to get space and time Order
    # The space order is the minimum space order
    # The time order is the minimum time_order
    for i in func:
        if isinstance(i, devito.Function):
            dtype = i.grid
            sp_dims += i.space_dimensions
            so = min(so, i.space_order)
            stagg = i.staggered
        if isinstance(i, devito.TimeFunction):
            dtype = i.grid
            sp_dims += i.space_dimensions
            time_dims += (i.time_dim,)
            so = min(so, i.space_order)
            is_time = True
            to = min(to, i.time_order)
            stagg = i.staggered

    obj.dtype = dtype or np.float32
    obj.space_order = so
    obj.time_order = to
    obj.space_dimensions = tuple(set(sp_dims))
    obj.staggered = stagg
    if is_time:
        obj.time_dim = tuple(set(time_dims))[0]
    obj.is_TimeFunction = is_time


class Pow(sympy.Mul, FrozenExpr):
    """A customized version of :class:`sympy.Add` representing a sum of
    symbolic object."""
    def __new__(cls, *args, **kwargs):
        return sympy.Pow.__new__(cls, *args, evaluate=False)

    def __init__(self, *args, **kwargs):
        fd_parameters(self)
        devito.finite_differences.finite_difference.initialize_derivatives(self)

    def __add__(self, other):
        return Add(self, other)

    def __iadd__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __imul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    @property
    def _time_size(self):
        func = list(retrieve_functions(self))
        for i in func:
            if isinstance(i, devito.TimeFunction):
                return i._time_size

        return None


class Mul(sympy.Mul, FrozenExpr):
    """A customized version of :class:`sympy.Add` representing a sum of
    symbolic object."""
    is_Mul = True

    def __new__(cls, *args, **kwargs):
        return sympy.Mul.__new__(cls, *args, evaluate=False)

    def __init__(self, *args, **kwargs):
        fd_parameters(self)
        devito.finite_differences.finite_difference.initialize_derivatives(self)

    def __add__(self, other):
        return Add(self, other)

    def __iadd__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __imul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    @property
    def _time_size(self):
        func = list(retrieve_functions(self))
        for i in func:
            if isinstance(i, devito.TimeFunction):
                return i._time_size

        return None


class Add(sympy.Add, FrozenExpr):
    """A customized version of :class:`sympy.Add` representing a sum of
    symbolic object."""
    is_Add = True

    def __new__(cls, *args, **kwargs):
        return sympy.Add.__new__(cls, *args, evaluate=False)

    def __init__(self, *args, **kwargs):
        fd_parameters(self)
        devito.finite_differences.finite_difference.initialize_derivatives(self)

    def __add__(self, other):
        return Add(self, other)

    def __iadd__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __imul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    @property
    def _time_size(self):
        func = list(retrieve_functions(self))
        for i in func:
            if isinstance(i, devito.TimeFunction):
                return i._time_size

        return None

    def evalf(self, N=None):
        N = N or sympy.N(sympy.Float(1.0))
        return Add(sum([a.evalf(N) for a in self.args]))

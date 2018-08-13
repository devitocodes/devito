import sympy
import numpy as np

import devito
from devito.symbolics.search import retrieve_functions
from devito.symbolics.extended_sympy import FrozenExpr

__all__ = ['Mul', 'Add', 'Pow', 'One', 'Zero']


class Differentiable(object):
    """
    This class represents Devito differentiable objects such as
    sum of functions, product of function or FD approximation and
    provides FD shortcuts for such expressions
    """

    def __init__(self, *args, **kwargs):
        self.space_order = kwargs.get('space_order', self.get_space_order)
        self.time_order = kwargs.get('time_order', self.get_time_order)
        self.dtype = kwargs.get('dtype', selg.get_dtype)
        fd_parameters(self)
        # Retrieve functions used in expression
        func = list(retrieve_functions(self))
        # Get FD parameters from the functions
        devito.finite_differences.finite_difference.initialize_derivatives(self)

    def __add__(self, other):
        return Add(self, other)

    def __iadd__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __imul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    @property
    def get_space_order(self):
        """
        Infer space_order from expression
        """
        func = list(retrieve_functions(obj))
        order = 100
        for i in func:
            order = min(order, i.space_order)

        return order

    @property
    def get_time_order(self):
        """
        Infer space_order from expression
        """
        func = list(retrieve_functions(obj))
        order = 100
        if i.is_TimeFunction for i in func:
            order = min(order, i.time_order)

        return order

    @property
    def get_dtype(self):
        """
        Infer dtype for expression
        """
        func = list(retrieve_functions(obj))
        dtype = np.float32
        for i in func:
            dtype = i.dtype if i.dtype == np.float64 else dtype

        return order

    def evalf(self, N=None):
        N = N or sympy.N(sympy.Float(1.0))
        return self.func(*[i.evalf(N) for i in self.args], evaluate=False)

One = Differentiable(1)
Zero = Differentiable(0)

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


class Pow(sympy.Mul, Differentiable):
    """A customized version of :class:`sympy.Add` representing a sum of
    symbolic object."""
    def __new__(cls, *args, **kwargs):
        return sympy.Pow.__new__(cls, *args, **kwargs)


class Mul(sympy.Mul, Differentiable):
    """A customized version of :class:`sympy.Add` representing a sum of
    symbolic object."""
    is_Mul = True

    def __new__(cls, *args, **kwargs):
        return sympy.Mul.__new__(cls, *args, **kwargs)


class Add(sympy.Add, Differentiable):
    """A customized version of :class:`sympy.Add` representing a sum of
    symbolic object."""
    is_Add = True

    def __new__(cls, *args, **kwargs):
        return sympy.Add.__new__(cls, *args, **kwargs)

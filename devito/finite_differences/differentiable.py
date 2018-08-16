import sympy
import numpy as np

import devito
from devito.tools.memoization import memoized_meth
from devito.logger import warning
from devito.symbolics.search import retrieve_functions
from devito.symbolics.extended_sympy import FrozenExpr

__all__ = ['Differentiable', 'Mul', 'Add', 'Pow', 'cos', 'sin']


class Differentiable(FrozenExpr):
    """
    This class represents Devito differentiable objects such as
    sum of functions, product of function or FD approximation and
    provides FD shortcuts for such expressions
    """
    is_Differentiable = True
    _op_priority = 100.0

    def __new__(cls, *args, **kwargs):
        return sympy.Expr.__new__(cls, *args)

    def __init__(self, *args, **kwargs):
        # Recover the list of possible FD shortcuts
        if kwargs.get('init', True):
            devito.finite_differences.finite_difference.initialize_derivatives(self)
            self.derivs = self.derivatives + (kwargs.get('derivs', ()),)

    def __add__(self, other):
        return Add(*[self, other])

    def __iadd__(self, other):
        return Add(*[self, other])

    def __radd__(self, other):
        return Add(*[self, other])

    def __sub__(self, other):
        return Add(*[self, -other])

    def __isub__(self, other):
        return Add(*[self, -other])

    def __rsub__(self, other):
        return Add(*[-self, other])

    def __mul__(self, other):
        return Mul(*[self, other])

    def __imul__(self, other):
        return Mul(*[self, other])

    def __rmul__(self, other):
        return Mul(*[self, other])

    @property
    def space_order(self):
        return self._space_order()

    @memoized_meth
    def _space_order(self):
        """
        Infer space_order from expression
        """
        func = list(retrieve_functions(self))
        order = 100
        for i in func:
            order = min(order, getattr(i, 'space_order', order))

        return order

    @property
    def time_order(self):
        return self._time_order()

    @memoized_meth
    def _time_order(self):
        """
        Infer space_order from expression
        """
        func = list(retrieve_functions(self))
        order = 100
        for i in func:
            order = min(order, getattr(i, 'time_order', order))

        return order

    @property
    def dtype(self):
        return self._dtype()

    @memoized_meth
    def _dtype(self):
        """
        Infer dtype for expression
        """
        func = list(retrieve_functions(self))
        is_double = False
        for i in func:
            dtype_i = getattr(i, 'dtype', np.float32)
            is_double = dtype_i == np.float64 or is_double

        return np.float64 if is_double else np.float32

    @property
    def indices(self):
        return self._indices()


    @memoized_meth
    def _indices(self):
        """
        Indices of the expression setup
        """
        func = list(retrieve_functions(self))
        return tuple(set([d for i in func for d in getattr(i, 'indices', ())]))

    @property
    def staggered(self):
        return self._staggered()

    @memoized_meth
    def _staggered(self):
        """
        Staggered grid setup
        """
        return tuple([None] * len(self.indices))

    def evalf(self, N=None):
        N = N or sympy.N(sympy.Float(1.0))
        if self.is_Number:
            return self.args[0]
        else:
            return self.func(*[i.evalf(N) for i in self.args], evaluate=False)

    def __getattr__(self, name):
        """
        Overload gettattr for derivatives as FD derivativees are linear
        Return sum of FD derivatives rather than creating new FD functions
        """
        if name == "derivs":
            raise AttributeError()
        if name in self.derivs:
            return self.getdiff(name)

        return self.__getattribute__(name)

    def getdiff(self, name):
        if name in self.derivs:
            return self.__getattribute__(name)
        else:
            warning("FD shortcut %s not found for Function %s " % (name, self) +
                    "and dimension %s, returning 0" % name[1])
            return 0

class Pow(Differentiable, sympy.Mul):
    """A customized version of :class:`sympy.Pow` representing a Power of
    symbolic object."""
    def __new__(cls, *args, **kwargs):
        return sympy.Pow.__new__(cls, *args, **kwargs)

class Mul(Differentiable, sympy.Mul):
    """A customized version of :class:`sympy.Mul` representing a product of
    symbolic object."""
    is_Mul = True

    def __new__(cls, *args, **kwargs):
        return sympy.Mul.__new__(cls, *args, **kwargs)


class Add(Differentiable, sympy.Add):
    """A customized version of :class:`sympy.Add` representing a sum of
    symbolic object."""
    is_Add = True

    def __new__(cls, *args, **kwargs):
        return sympy.Add.__new__(cls, *args, **kwargs)

def cos(function):
    return sympy.cos(function)


def sin(function):
    return sympy.sin(function)


def to_differentiable(expr):
    if getattr(expr, 'is_Differentiable', False):
        return expr
    elif expr.is_Add:
        return Add(*expr.args)
    elif expr.is_Mul:
        return Mul(*expr.args)
    elif expr.is_Pow:
        return Pow(*expr.args)
    else:
        return expr

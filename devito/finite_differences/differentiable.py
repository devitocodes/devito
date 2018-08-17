import sympy
import numpy as np

from devito.symbolics.search import retrieve_functions
from devito.symbolics.extended_sympy import FrozenExpr

__all__ = ['Differentiable']


class Differentiable(FrozenExpr):
    """
    This class represents Devito differentiable objects such as
    sum of functions, product of function or FD approximation and
    provides FD shortcuts for such expressions
    """
    _op_priority = 100.0

    def __new__(cls, *args, **kwargs):
        return sympy.Expr.__new__(cls, *args)

    def __init__(self, expr, **kwargs):
        from devito.finite_differences.finit_difference import generate_fd_functions
        self.expr = expr.expr if isinstance(expr, Differentiable) else expr
        # Recover the list of possible FD shortcuts
        self.dtype = self._dtype()
        self.space_order = self._space_order()
        self.time_order = self._space_order()
        self.indices = self._indices()
        self.staggered = self._staggered()
        # Generate FD shortcuts for expression or copy from input
        if isinstance(expr, Differentiable):
            self.fd = expr.fd
        else:
            self.fd = generate_fd_functions(self)

        for d in self.fd:
            setattr(self.__class__, d[1], property(d[0], d[1]))
        self.derivatives = tuple(d[1] for d in self.fd)

    def __add__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(sympy.Add(*[self.expr, other.expr]))
        else:
            return Differentiable(sympy.Add(*[self.expr, other]))

    __iadd__ = __add__
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(sympy.Add(*[self.expr, -other.expr]))
        else:
            return Differentiable(sympy.Add(*[self.expr, -other]))

    def __rsub__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(sympy.Add(*[-self.expr, other.expr]))
        else:
            return Differentiable(sympy.Add(*[-self.expr, other]))

    __isub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(sympy.Mul(*[self.expr, other.expr]))
        else:
            return Differentiable(sympy.Mul(*[self.expr, other]))

    __imul__ = __mul__
    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Differentiable):
            return Differentiable(self.expr/other.expr)
        else:
            return Differentiable(self.expr/other)

    def func(self, *args, **kwargs):
        return Differentiable(self.expr.func(*args), **kwargs)

    def __str__(self):
        return self.expr.__str__()

    __repr__ = __str__

    @property
    def args(self):
        return self.expr.args

    def _space_order(self):
        """
        Infer space_order from expression
        """
        func = list(retrieve_functions(self.expr))
        order = 100
        for i in func:
            order = min(order, getattr(i, 'space_order', order))

        return order

    def _time_order(self):
        """
        Infer space_order from expression
        """
        func = list(retrieve_functions(self.expr))
        order = 100
        for i in func:
            order = min(order, getattr(i, 'time_order', order))

        return order

    def _dtype(self):
        """
        Infer dtype for expression
        """
        func = list(retrieve_functions(self.expr))
        is_double = False
        for i in func:
            dtype_i = getattr(i, 'dtype', np.float32)
            is_double = dtype_i == np.float64 or is_double

        return np.float64 if is_double else np.float32

    def _indices(self):
        """
        Indices of the expression setup
        """
        func = list(retrieve_functions(self.expr))
        return tuple(set([d for i in func for d in getattr(i, 'indices', ())]))

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

    def subs(self, subs):
        return Differentiable(self.expr.subs(subs))

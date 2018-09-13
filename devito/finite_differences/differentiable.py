import sympy
import numpy as np

from devito.tools import filter_ordered
from devito.finite_differences.utils import to_expr

__all__ = ['Differentiable']


class Differentiable(sympy.Expr):
    """
    This class represents Devito differentiable objects such as
    sum of functions, product of function or FD approximation and
    provides FD shortcuts for such expressions
    """
    # Set the operator priority higher than Sympy (10.0) to force the overloaded
    # operators to be used
    _op_priority = 100.0

    def __new__(cls, *args, **kwargs):
        if cls == Differentiable:
            assert len(args) == 1
            expr = args[0]
            if expr.is_Function:
                return expr
            new_obj = sympy.Expr.__new__(cls, expr)
            # Initialization
            new_obj.__init__(expr, **kwargs)
            return new_obj
        return sympy.Expr.__new__(cls, *args)

    def __init__(self, expr, **kwargs):
        # Set FD properties from input
        self._dtype = kwargs.get('dtype')
        self._space_order = kwargs.get('space_order')
        self._time_order = kwargs.get('time_order')
        self._indices = kwargs.get('indices', ())
        self._staggered = kwargs.get('staggered')
        self._grid = kwargs.get('grid')
        # Generate FD shortcuts for expression or copy from input
        self._fd = kwargs.get('fd', {})
        # Associated Sympy expression
        self._expr = to_expr(expr)

    @property
    def space_order(self):
        return self._space_order

    @property
    def time_order(self):
        return self._time_order

    @property
    def staggered(self):
        return self._staggered

    @property
    def indices(self):
        return self._indices

    @property
    def fd(self):
        return self._fd

    @property
    def grid(self):
        return self._grid

    @property
    def dtype(self):
        return self._dtype

    def __getattr__(self, name):
        if name == 'fd' or name == '_fd':
            raise AttributeError()
        if name in self.fd:
            # self.fd[name] = (property, description), calls self.fd[name][0]
            return self.fd[name][0](self)
        return self.__getattribute__(name)

    def xreplace(self, rule):
        if self.is_Function:
            return super(Differentiable, self).xreplace(rule)
        else:
            return self._expr.xreplace(rule)

    def _merge_fd_properties(self, other):
        merged = {}
        merged["space_order"] = np.min([getattr(self, 'space_order', 100) or 100,
                                        getattr(other, 'space_order', 100)])
        merged["time_order"] = np.min([getattr(self, 'time_order', 100) or 100,
                                       getattr(other, 'time_order', 100)])
        merged["indices"] = tuple(filter_ordered(self.indices +
                                                 getattr(other, 'indices', ())))
        merged["fd"] = dict(getattr(self, 'fd', {}), **getattr(other, 'fd', {}))
        merged["staggered"] = self.staggered
        merged["dtype"] = self.dtype
        return merged

    @property
    def args(self):
        if self.is_Function:
            return super(Differentiable, self).args
        return (self._expr,)

    def __add__(self, other):
        return Differentiable(sympy.Add(*[getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)]),
                              **self._merge_fd_properties(other))

    def __iadd__(self, other):
        self._expr = sympy.Add(*[getattr(self, '_expr', self),
                                 getattr(other, '_expr', other)])

        return self

    __radd__ = __add__

    def __sub__(self, other):
        return Differentiable(sympy.Add(*[getattr(self, '_expr', self),
                                          -getattr(other, '_expr', other)]),
                              **self._merge_fd_properties(other))

    def __rsub__(self, other):
        return Differentiable(sympy.Add(*[-getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)]),
                              **self._merge_fd_properties(other))

    def __isub__(self, other):
        self._expr = sympy.Add(*[getattr(self, '_expr', self),
                                 -getattr(other, '_expr', other)])
        return self

    def __mul__(self, other):
        return Differentiable(sympy.Mul(*[getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)]),
                              **self._merge_fd_properties(other))

    def __imul__(self, other):
        self._expr = sympy.Mul(*[getattr(self, '_expr', self),
                                 getattr(other, '_expr', other)])
        return self

    __rmul__ = __mul__

    def __truediv__(self, other):
        return Differentiable(getattr(self, '_expr', self) *
                              (getattr(other, '_expr', other) ** (-1)),
                              **self._merge_fd_properties(other))

    def __rtruediv__(self, other):
        return Differentiable(getattr(other, '_expr', other) *
                              (getattr(self, '_expr', self) ** (-1)),
                              **self._merge_fd_properties(other))

    __floordiv__ = __truediv__
    __rdiv__ = __rtruediv__
    __div__ = __truediv__
    __rfloordiv__ = __rtruediv__

    def __pow__(self, other):
        if other > 0:
            return Differentiable(sympy.Mul(*[getattr(self, '_expr', self)]*other,
                                            evaluate=False),
                                  **self._merge_fd_properties(None))
        elif other < 0:
            return Differentiable(sympy.Pow(*[getattr(self, '_expr', self), other]),
                                  **self._merge_fd_properties(None))
        else:
            return sympy.Number(1)

    def __rpow__(self, other):
        return other.__pow__(self)

    def __neg__(self):
        return Differentiable(sympy.Mul(*[getattr(self, '_expr', self), -1]),
                              **self._merge_fd_properties(None))

    def __str__(self):
        if self.is_Function:
            return super(Differentiable, self).__str__()
        return self._expr.__str__()

    __repr__ = __str__

    def __eq__(self, other):
        expr = getattr(self, '_expr', self)
        oth = getattr(other, '_expr', other)
        if expr.is_Function:
            return super(Differentiable, self).__eq__(oth)
        return expr.__eq__(oth)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __int__(self):
        if self.is_Function:
            return super(Differentiable, self).__int__()
        return Differentiable(self._expr.__int__(), **self._merge_fd_properties(None))

    def __float__(self):
        if self.is_Function:
            return super(Differentiable, self).__float__()
        return Differentiable(self._expr.__float__(), **self._merge_fd_properties(None))

    def __mod__(self, other):
        if self.is_Function:
            return super(Differentiable, self).__fmode__(other)
        return Differentiable(self._expr.__mod__(getattr(other, '_expr', other)),
                              **self._merge_fd_properties(None))

    def __rmod__(self, other):
        return other.__mod__(self)

    def subs(self, *subs):
        if self.is_Function:
            return super(Differentiable, self).subs(*subs)
        expr_sub = to_expr(self._expr).subs(*subs)
        return Differentiable(expr_sub, **self._merge_fd_properties(None))

    def __hash__(self):
        return hash(self._expr)

    def _hashable_content(self):
        if self.is_Function:
            return super(Differentiable, self)._hashable_content()
        return self._expr._hashable_content()

    @property
    def laplace(self):
        """
        Generates a symbolic expression for the Laplacian, the second
        derivative wrt. all spatial dimensions.
        """
        space_dims = [d for d in self.indices if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
        return sum([getattr(self, d) for d in derivs])

    def laplace2(self, weight=1):
        """
        Generates a symbolic expression for the double Laplacian
        wrt. all spatial dimensions.
        """
        space_dims = [d for d in self.indices if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
        return sum([getattr(self.laplace * weight, d) for d in derivs])

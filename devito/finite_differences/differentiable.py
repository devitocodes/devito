import sympy
import numpy as np

from devito.symbolics.extended_sympy import ExprDiv
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
        self._expr = expr

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

    def __add__(self, other):
        return Differentiable(sympy.Add(*[getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)]),
                              **self._merge_fd_properties(other))

    __iadd__ = __add__
    __radd__ = __add__

    def __sub__(self, other):
        return Differentiable(sympy.Add(*[getattr(self, '_expr', self),
                                          -getattr(other, '_expr', other)]),
                              **self._merge_fd_properties(other))

    def __rsub__(self, other):
        return Differentiable(sympy.Add(*[-getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)]),
                              **self._merge_fd_properties(other))

    __isub__ = __sub__

    def __mul__(self, other):
        return Differentiable(sympy.Mul(*[getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)]),
                              **self._merge_fd_properties(other))

    __imul__ = __mul__
    __rmul__ = __mul__

    def __truediv__(self, other):
        return Differentiable(sympy.Mul(*[getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)**(-1)]),
                              **self._merge_fd_properties(other))

    def __rtruediv__(self, other):
        return Differentiable(sympy.Mul(*[getattr(other, '_expr', other),
                                          getattr(self, '_expr', self)**(-1)]),
                              **self._merge_fd_properties(other))

    __floordiv__ = __truediv__
    __rdiv__ = __rtruediv__
    __div__ = __truediv__
    __rfloordiv__ = __rtruediv__

    def __pow__(self, exponent):
        if exponent > 0:
            return Differentiable(sympy.Mul(*[getattr(self, '_expr', self)]*exponent,
                                            evaluate=False),
                                  **self._kwargs)
        elif exponent < 0:
            return Differentiable(ExprDiv(sympy.Number(1),
                                          sympy.Mul(*[getattr(self, '_expr',
                                                      self)]*(-exponent),
                                                    evaluate=False)),
                                  **self._merge_fd_properties(None))
        else:
            return sympy.Number(1)

    def __neg__(self):
        return Differentiable(sympy.Mul(*[getattr(self, '_expr', self), -1]),
                              **self._merge_fd_properties(None))

    def __str__(self):
        if self.is_Function:
            return super(Differentiable, self).__str__()
        return to_expr(self._expr).__str__()

    __repr__ = __str__

    def subs(self, *subs):
        if self.is_Function:
            return super(Differentiable, self).subs(*subs)
        return self._expr.subs(*subs)

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

import sympy
from sympy.core.basic import _aresame
import numpy as np

from devito.finite_differences.finite_difference import generate_fd_functions
from devito.symbolics.extended_sympy import ExprDiv
from devito.finite_differences.utils import to_expr

__all__ = ['Differentiable']


class Differentiable(sympy.Expr):
    """
    This class represents Devito differentiable objects such as
    sum of functions, product of function or FD approximation and
    provides FD shortcuts for such expressions
    """
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
        # Recover the list of possible FD shortcuts
        self.dtype = kwargs.get('dtype')
        self.space_order =  kwargs.get('space_order')
        self.time_order =  kwargs.get('time_order', 0)
        self.indices =  kwargs.get('indices', ())
        self.staggered =  kwargs.get('staggered')
        self.grid = kwargs.get('grid')
        # Generate FD shortcuts for expression or copy from input
        self.fd = kwargs.get('fd', [])
        for d in self.fd:
            setattr(self.__class__, d[1], property(d[0], d[1]))
        self.derivatives = tuple(d[1] for d in self.fd)
        # Save kwargs
        self._kwargs = kwargs
        self._expr = expr

    def xreplace(self, rule):
        out = getattr(self, '_expr', self)
        if out in rule:
            return rule[out]
        elif rule:
            args = []
            for a in out.args:
                try:
                    args.append(a.xreplace(rule))
                except AttributeError:
                    args.append(a)
            args = tuple(args)
            if not _aresame(args, out.args):
                return out.func(*args)
        return out

    def merge_fd_properties(self, other):
        merged = getattr(self, '_kwargs', dict())
        merged["space_order"] = np.min([getattr(self, 'space_order', 100) or 100 ,
                                        getattr(other, 'space_order', 100)])
        merged["time_order"] = np.min([getattr(self, 'time_order', 100) or 100,
                                       getattr(other, 'time_order', 100)])
        merged["indices"] = tuple(set(self.indices + getattr(other, 'indices', ())))
        merged["fd"] = list(set(getattr(self, 'fd', []) + getattr(other, 'fd', [])))
        merged["staggered"] = self.staggered
        merged["dtype"] = self.dtype
        return merged

    def __add__(self, other):
        return Differentiable(sympy.Add(*[getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)]),
                                          **self.merge_fd_properties(other))

    __iadd__ = __add__
    __radd__ = __add__

    def __sub__(self, other):
        return Differentiable(sympy.Add(*[getattr(self, '_expr', self),
                                          -getattr(other, '_expr', other)]),
                                          **self.merge_fd_properties(other))

    def __rsub__(self, other):
        return Differentiable(sympy.Add(*[-getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)]),
                                          **self.merge_fd_properties(other))

    __isub__ = __sub__

    def __mul__(self, other):
        return Differentiable(sympy.Mul(*[getattr(self, '_expr', self),
                                          getattr(other, '_expr', other)]),
                                          **self.merge_fd_properties(other))

    __imul__ = __mul__
    __rmul__ = __mul__

    def __truediv__(self, other):
        return Differentiable(sympy.Mul(*[getattr(self, '_expr', self), other**(-1)]),
                                        **self.merge_fd_properties(other))

    def __rtruediv__(self, other):
        return Differentiable(sympy.Mul(*[other, getattr(self, '_expr', self)**(-1)]),
                                        **self.merge_fd_properties(other))

    def __pow__(self, exponent):
        if exponent > 0:
            return Differentiable(sympy.Mul(*[getattr(self, '_expr', self)]*exponent, evaluate=False),**self._kwargs)
        elif exponent < 0 :
            return Differentiable(ExprDiv(sympy.Number(1),
                                          sympy.Mul(*[getattr(self, '_expr', self)]*(-exponent),
                                                    evaluate=False)),
                                  **self._kwargs)
        else:
            return sympy.Number(0)

    def __str__(self):
        if self.is_Function:
            return super(sympy.Expr, self).__str__()
        return self._expr.__str__()

    __repr__ = __str__

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

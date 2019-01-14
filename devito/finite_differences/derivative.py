from cached_property import cached_property

from devito.finite_differences.finite_difference import left, right, centered, generic_derivative, first_derivative, cross_derivative
from devito.finite_differences.differentiable import Differentiable

__all__ = ['Derivative']

class Derivative(object):

    """
    Represents an unevaluated  derivative of an input expression
    """

    def __init__(self, expr, deriv_order, dims, fd_order, **kwargs):
        self._expr = expr
        self._dims = dims
        self._fd_order = fd_order
        self._deriv_order = deriv_order
        self._stagger = kwargs.get("stagger", self._stagger_setup)
        self._side = kwargs.get("side", None)

    @cached_property
    def expr(self):
        return self._expr

    @cached_property
    def dims(self):
        return self._dims

    @cached_property
    def fd_order(self):
        return self._fd_order

    @cached_property
    def deriv_order(self):
        return self._deriv_order

    @cached_property
    def stagger(self):
        return self._stagger

    @cached_property
    def side(self):
        return self._side


    @cached_property
    def _stagger_setup(self):
        if not self.expr.is_Staggered:
            side = None
        else:
            dims = self.expr.indices
            side = dict()
            for (d, s) in zip(dims, self.expr.staggered):
                if s == 0:
                    side[d] = left
                elif s == 1:
                    side[d] = right
                else:
                    side[d] = centered

        return side

    @cached_property
    def stencil(self):
        if self.side is not None:
            return first_derivative(self.expr.stencil, self.dims, self.fd_order,
                                    side=self.side)
        if isinstance(self.dims, tuple):
            return cross_derivative(self.expr.stencil, self.dims, self.fd_order,
                                    self.deriv_order, stagger=self.stagger)
        else:
            return generic_derivative(self.expr.stencil, self.dims, self.fd_order,
                                      self.deriv_order, stagger=self.stagger)


    def __repr__(self):
        return "d^%s/d%s^%s (%s)" % (self.deriv_order, self.dims,
                                     self.deriv_order, self.expr)

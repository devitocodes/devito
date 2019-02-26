import sympy

from cached_property import cached_property

from devito.finite_differences.finite_difference import (left, right, centered,
                                                         generic_derivative, transpose,
                                                         first_derivative, direct,
                                                         cross_derivative)
from devito.finite_differences.differentiable import Differentiable
from devito.tools import as_tuple
from devito.types import Dimension

__all__ = ['Diff']


class Diff(sympy.Derivative, Differentiable):

    """
    Represents an unevaluated  derivative of an input expression
    """

    def __new__(cls, expr, *dims, **kwargs):
        deriv_order = kwargs.get("deriv_order", 1)
        # expand the dimension depending on the derivative order
        # ie Diff(expr, x, 2) becomes Derivative(expr, (x, 2))
        if not isinstance(deriv_order, tuple):
            new_dims = tuple([dims[0] for _ in range(deriv_order)])
        else:
            new_dims = []
            for d, o in zip(*dims, deriv_order):
                for _ in range(o):
                    new_dims.append(d)

            new_dims = tuple(new_dims)

        new_obj = sympy.Derivative.__new__(cls, expr, *new_dims)
        new_obj.setup_fd(expr, *dims, **kwargs)
        return new_obj

    def setup_fd(self, expr, dims, deriv_order=1, fd_order=1, **kwargs):
        self._dims = self._setup_wrt(dims)
        self._fd_order = fd_order
        self._deriv_order = deriv_order
        self._stagger = kwargs.get("stagger", self._stagger_setup)
        self._side = kwargs.get("side", None)
        self._transpose = kwargs.get("transpose", direct)

    def _setup_wrt(self, dims):
        dims = as_tuple(dims)
        return tuple(d for d in dims if isinstance(d, Dimension))

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
    def is_Staggered(self):
        return self.expr.is_Staggered

    @cached_property
    def indices(self):
        return self.expr.indices

    @cached_property
    def staggered(self):
        return self.expr.staggered

    @cached_property
    def _stagger_setup(self):
        if not self.is_Staggered:
            side = dict((d, None) for d in self.dims)
        else:
            dims = self.indices
            side = dict()
            for (d, s) in zip(dims, self.staggered):
                if s == 0:
                    side[d] = left
                elif s == 1:
                    side[d] = right
                else:
                    side[d] = centered

        return side

    @property
    def transpose(self):
        return self._transpose

    @property
    def T(self):
        """transpose of the derivative"""
        if self._transpose == direct:
            adjoint = transpose
        else:
            adjoint = direct

        return Diff(self.expr, *self.dims, deriv_order=self.deriv_order,
                    fd_order=self.fd_order, side=self.side, stagger=self.stagger,
                    transpose=adjoint)

    @property
    def stencil(self):
        expr = getattr(self.expr, 'stencil', self.expr)
        if self.side in [left, right] and self.deriv_order == 1:
            res = first_derivative(expr, self.dims[0], self.fd_order,
                                   side=self.side, matvec=self.transpose)
        elif len(self.dims) > 1:
            res = cross_derivative(expr, self.dims, self.fd_order, self.deriv_order,
                                   matvec=self.transpose, stagger=self.stagger)
        else:
            res = generic_derivative(expr, self.dims[0], self.fd_order,
                                     self.deriv_order, stagger=self.stagger[self.dims[0]],
                                     matvec=self.transpose)
        return res

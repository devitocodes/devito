import sympy

from cached_property import cached_property

from devito.finite_differences.finite_difference import left, right, centered, generic_derivative, first_derivative, cross_derivative
from devito.finite_differences.differentiable import Differentiable

__all__ = ['Diff']

class Diff(sympy.Derivative, Differentiable):

    """
    Represents an unevaluated  derivative of an input expression
    """

    def __new__(cls, expr, *dims, **kwargs):
        new_obj = sympy.Derivative.__new__(cls, expr, *dims)
        new_obj.setup_fd(expr, *dims, **kwargs)
        return new_obj

    def setup_fd(self, expr, *dims, deriv_order=1, fd_order=1, **kwargs):
        self._dims = dims
        self._fd_order = fd_order
        self._deriv_order = deriv_order
        self._stagger = kwargs.get("stagger", self._stagger_setup)

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

    @cached_property
    def indices(self):
        indices = dict()
        for d in self.dims:
            diff = d.spacing
            stagger = self.stagger[d] or self.stagger
            if stagger is None:
                off = 0
            elif stagger == left or not self.expr.is_Staggered:
                off = -.5
            elif stagger == right:
                off = .5
            else:
                off = 0

            if self.expr.is_Staggered:
                indices[d] = list(set([(d + int(i+.5+off) * d.spacing)
                                       for i in range(-self.fd_order//2, self.fd_order//2)]))
                if fd_order < 2:
                    indices[d] = [d + diff, d] if stagger == right else [d - diff, d]

            else:
                indices[d] = [(d + i * diff) for i in range(-self.fd_order//2, self.fd_order//2 + 1)]

            if self.fd_order < 2:
                indices[d] = [d, d + diff]

        return list(indices[d] for d in self.dims)


    @cached_property
    def x0(self):
        x0 = dict()
        for d in self.dims:
            diff = d.spacing
            stagger = self.stagger[d] or self.stagger
            if stagger is None:
                off = 0
            elif stagger == left or not self.expr.is_Staggered:
                off = -.5
            elif stagger == right:
                off = .5
            else:
                off = 0
            if self.expr.is_Staggered:
                x0[d] = (d + off*diff)
            else:
                x0[d] = d
        return list(x0[d] for d in self.dims)

    @cached_property
    def stencil(self):
        # if self.side is not None:
        #     return first_derivative(self.expr.stencil, self.dims[0], self.fd_order,
        #                             side=self.side)
        # if isinstance(self.dims, tuple):
        #     return cross_derivative(self.expr.stencil, self.dims, self.fd_order,
        #                             self.deriv_order, stagger=self.stagger)
        # else:
        #     return generic_derivative(self.expr.stencil, self.dims[0], self.fd_order,
        #                               self.deriv_order, stagger=self.stagger)
        res = self
        for d, x0, i in zip(self.dims, self.x0, self.indices):
            res = res.as_finite_difference(i, x0=x0, wrt=d)
        return res

import sympy

from cached_property import cached_property

from devito.tools import flatten
from devito.finite_differences.finite_difference import (left, right, centered,
                                                         generic_derivative, transpose,
                                                         first_derivative, direct,
                                                         cross_derivative)
from devito.finite_differences.differentiable import Differentiable
from devito.tools import as_tuple
from devito.types import Dimension


class Derivative(sympy.Derivative, Differentiable):

    """
    An unevaluated Derivative, which carries metadata (Dimensions,
    derivative order, etc) to be evaluated.
    """

    def __new__(cls, expr, *dims, **kwargs):
        print(expr, dims, kwargs)
        # Verifies that there is one order per dimension if
        # deriv_order is provided
        deriv_order = kwargs.get('deriv_order', None)
        ndim = len(as_tuple(dims))
        norder = len(as_tuple(deriv_order)) if deriv_order is not None else 0

        if deriv_order is not None and ndim != norder:
            raise ValueError("Different number of Dimensions and derivative orders")
    
        if deriv_order is not None:
            orders = dict((d, o) for d, o in zip(dims, as_tuple(deriv_order)))
        else:
            orders = dict()
            for i, d in enumerate(dims):
                if isinstance(d, (tuple, sympy.Tuple)):
                    orders[d[0]] = d[1]
                elif int(d) == d:
                    orders[dims[i-1]] = d
                elif d in orders:
                    orders[d] =+ 1
                else:
                    orders[d] = 1

        new_dims = []
        for k, v in orders.items():
            new_dims += [k for _ in range(v)]
        new_dims = tuple(new_dims)

        obj = sympy.Derivative.__new__(cls, expr, *new_dims, evaluate=False)
        obj._dims = new_dims
        obj._fd_order = kwargs.get('fd_order', 1)
        obj._deriv_order = tuple(orders.values())
        obj._stagger = kwargs.get("stagger", obj._stagger_setup)
        obj._side = kwargs.get("side", None)
        obj._transpose = kwargs.get("transpose", direct)
        
        return obj

    @property
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
        """Transpose of the Derivative."""
        if self._transpose == direct:
            adjoint = transpose
        else:
            adjoint = direct

        return Derivative(self.expr, *self.dims, deriv_order=self.deriv_order,
                          fd_order=self.fd_order, side=self.side, stagger=self.stagger,
                          transpose=adjoint)

    @property
    def evaluate(self):
        expr = getattr(self.expr, 'evaluate', self.expr)
        if self.side in [left, right] and self.deriv_order == 1:
            res = first_derivative(expr, self.dims[0], self.fd_order,
                                   side=self.side, matvec=self.transpose)
        elif len(self.dims) > 1:
            res = cross_derivative(expr, self.dims, self.fd_order, self.deriv_order,
                                   matvec=self.transpose, stagger=self.stagger)
        else:
            res = generic_derivative(expr, *self.dims, self.fd_order,
                                     self.deriv_order, stagger=self.stagger,
                                     matvec=self.transpose)
        return res

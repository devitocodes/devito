from collections import Iterable, OrderedDict

import sympy
from cached_property import cached_property

from devito.finite_differences.finite_difference import (generic_derivative,
                                                         first_derivative,
                                                         cross_derivative)
from devito.finite_differences.differentiable import Differentiable
from devito.finite_differences.tools import centered, direct, transpose, left, right
from devito.tools import as_tuple, filter_ordered

__all__ = ['Derivative']


class Derivative(sympy.Derivative, Differentiable):

    """
    An unevaluated Derivative, which carries metadata (Dimensions,
    derivative order, etc) describing how the derivative will be expanded
    upon evaluation.

    Parameters
    ----------
    expr : expr-like
        Expression for which the Derivative is produced.
    dims : Dimension or tuple of Dimension
        Dimenions w.r.t. which to differentiate.
    fd_order : int or tuple of int, optional
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil. Defaults to 1.
    deriv_order: int or tuple of int, optional
        Derivative order. Defaults to 1.
    stagger : Side or tuple of Side, optional
        Shift of the finite-difference approximation. Defaults to ``centered``.
    side : Side or tuple of Side, optional
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1). Defaults to ``centered``.
    transpose : Transpose, optional
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference. Defaults to ``direct``.

    Examples
    --------
    Creation

    >>> from devito import Function, Derivative, Grid
    >>> grid = Grid((10, 10))
    >>> x, y = grid.dimensions
    >>> u = Function(name="u", grid=grid, space_order=2)
    >>> Derivative(u, x)
    Derivative(u(x, y), x)

    This can also be obtained via the differential shortcut

    >>> u.dx
    Derivative(u(x, y), x)

    You can also specify the order as a keyword argument

    >>> Derivative(u, x, deriv_order=2)
    Derivative(u(x, y), (x, 2))

    Or as a tuple

    >>> Derivative(u, (x, 2))
    Derivative(u(x, y), (x, 2))

    Once again, this can be obtained via shortcut notation

    >>> u.dx2
    Derivative(u(x, y), (x, 2))
    """

    _state = ('expr', 'dims', 'side', 'stagger', 'fd_order', 'transpose', '_eval_at')

    def __new__(cls, expr, *dims, **kwargs):
        if type(expr) == sympy.Derivative:
            raise ValueError("Cannot nest sympy.Derivative with devito.Derivative")
        if not isinstance(expr, Differentiable):
            raise ValueError("`expr` must be a Differentiable object")

        # Check `dims`. It can be a single Dimension, an iterable of Dimensions, or even
        # an iterable of 2-tuple (Dimension, deriv_order)
        if len(dims) == 0:
            raise ValueError("Expected Dimension w.r.t. which to differentiate")
        elif len(dims) == 1:
            if isinstance(dims[0], Iterable):
                # Iterable of Dimensions
                if len(dims[0]) != 2:
                    raise ValueError("Expected `(dim, deriv_order)`, got %s" % dims[0])
                orders = kwargs.get('deriv_order', dims[0][1])
                if dims[0][1] != orders:
                    raise ValueError("Two different values of `deriv_order`")
                new_dims = tuple([dims[0][0]]*dims[0][1])
            else:
                # Single Dimension
                orders = kwargs.get('deriv_order', 1)
                if isinstance(orders, Iterable):
                    orders = orders[0]
                new_dims = tuple([dims[0]]*orders)
        else:
            # Iterable of 2-tuple, e.g. ((x, 2), (y, 3))
            new_dims = []
            orders = []
            d_ord = kwargs.get('deriv_order', tuple([1]*len(dims)))
            for d, o in zip(dims, d_ord):
                if isinstance(d, Iterable):
                    new_dims.extend([d[0] for _ in range(d[1])])
                    orders.append(d[1])
                else:
                    new_dims.extend([d for _ in range(o)])
                    orders.append(o)
            new_dims = as_tuple(new_dims)
            orders = as_tuple(orders)

        # Finite difference orders depending on input dimension (.dt or .dx)
        fd_orders = kwargs.get('fd_order', tuple([expr.time_order if
                                                  getattr(d, 'is_Time', False) else
                                                  expr.space_order for d in dims]))
        if len(dims) == 1 and isinstance(fd_orders, Iterable):
            fd_orders = fd_orders[0]

        # SymPy expects the list of variable w.r.t. which we differentiate to be a list
        # of 2-tuple `(s, count)` where s is the entity to diff wrt and count is the order
        # of the derivative
        variable_count = [sympy.Tuple(s, new_dims.count(s))
                          for s in filter_ordered(new_dims)]

        # Construct the actual Derivative object
        obj = Differentiable.__new__(cls, expr, *variable_count)
        obj._dims = tuple(OrderedDict.fromkeys(new_dims))
        obj._fd_order = fd_orders
        obj._deriv_order = orders
        obj._side = kwargs.get("side", centered)
        obj._stagger = kwargs.get("stagger", tuple([left]*len(obj._dims)))
        obj._transpose = kwargs.get("transpose", direct)
        obj._eval_at = as_tuple(kwargs.get("eval_at"))

        return obj

    def _xreplace(self, eval_at):
        """
        This is a helper method used internally by SymPy. We exploit it to postpone
        substitutions until evaluation.
        """
        eval_at = self._eval_at + (eval_at,)  # Postponed substitutions
        return Derivative(self.expr, *self.dims, deriv_order=self.deriv_order,
                          fd_order=self.fd_order, side=self.side, stagger=self.stagger,
                          transpose=self.transpose, eval_at=eval_at), True

    @cached_property
    def _args_diff(self):
        return tuple(i for i in self.args if isinstance(i, Differentiable))

    @property
    def dims(self):
        return self._dims

    @property
    def fd_order(self):
        return self._fd_order

    @property
    def deriv_order(self):
        return self._deriv_order

    @property
    def stagger(self):
        return self._stagger

    @property
    def side(self):
        return self._side

    @property
    def transpose(self):
        return self._transpose

    @property
    def T(self):
        """Transpose of the Derivative.

        FD derivatives can be represented as matrices and have adjoint/transpose.
        This is really useful for more advanced FD definitions. For example
        the conventional Laplacian is `.dxl.T * .dxl`
        """
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
                                     self.deriv_order, stagger=self.stagger[0],
                                     matvec=self.transpose)
        for e in self._eval_at:
            res = res.xreplace(e)
        return res

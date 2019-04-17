import sympy

from collections import OrderedDict
from devito.finite_differences.finite_difference import (generic_derivative,
                                                         first_derivative,
                                                         cross_derivative)
from devito.finite_differences.differentiable import Differentiable
from devito.finite_differences.tools import centered, direct, transpose, left, right
from devito.tools import as_tuple


class Derivative(sympy.Derivative, Differentiable):

    """
    An unevaluated Derivative, which carries metadata (Dimensions,
    derivative order, etc) to be evaluated.

    Parameters
    ----------

    expr : symbolic expression
    dims : Dimension or tuple
        Dimenions wrt which to take the derivative
    deriv_order: Int or Tuple
        Order of the derivative for each Dimension
    fd_order : Int or tuple
        Order of the finite difference for each Dimension
    stagger : Dimension or tuple
        Staggering for each dimension
    side : Side or tuple
        Side of the finite difference for each Dimension
    transpose : direct or adjoint
        Wether the finite difference is transposed

    Examples
    --------
    Creation

    >> from devito import Function, Derivative, Grid
    >> grid = Grid((10, 10))
    >> u = Function(name="u", grid=grid, space_order=2)
    >> Derivative(u, u.indices[0])
    Derivative(u(x, y), x)

    This can also be obtained via the differential shortcut

    >> u.dx

    For higher order you can specify the order as a keyword argument

    >> Derivative(u, x, deriv_order=2)
    Derivative(u(x, y), (x, 2))

    Or as a tuple

    >> Derivative(u, (x, 2))
    Derivative(u(x, y), (x, 2))

    Once again, this can be access via the differential shortcut

    >> u.dx2
    """

    _state = ('expr', 'dims', 'side', 'stagger', 'fd_order', 'transpose')

    def __new__(cls, expr, *dims, **kwargs):
        # Check dims, can be a dimensions, multiple dimensions as a tuple
        # or a tuple of tuple (ie ((x,1),))
        if len(dims) == 1:
            if isinstance(dims[0], (tuple, sympy.Tuple)):
                orders = kwargs.get('deriv_order', dims[0][1])
                if dims[0][1] != orders:
                    raise ValueError("Two different value of deriv_order")
                new_dims = tuple([dims[0][0]]*dims[0][1])
            else:
                orders = kwargs.get('deriv_order', 1)
                new_dims = tuple([dims[0]]*orders)
        else:
            # ie ((x, 2), (y, 3))
            new_dims = []
            orders = []
            for d in dims:
                if isinstance(d, (tuple, sympy.Tuple)):
                    new_dims += [d[0] for _ in range(d[1])]
                    orders += [d[1]]
                else:
                    new_dims += [d]
                    orders += [1]
            new_dims = as_tuple(new_dims)

        kwargs["evaluate"] = False
        kwargs["simplify"] = False
        obj = sympy.Derivative.__new__(cls, expr, *new_dims, **kwargs)
        obj._dims = tuple(OrderedDict.fromkeys(new_dims))
        obj._fd_order = kwargs.get('fd_order', 1)
        obj._deriv_order = orders
        obj._side = kwargs.get("side", None)
        obj._stagger = kwargs.get("stagger", tuple([centered]*len(obj._dims)))
        obj._transpose = kwargs.get("transpose", direct)

        return obj

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
    def is_Staggered(self):
        return self.expr.is_Staggered

    @property
    def indices(self):
        return self.expr.indices

    @property
    def staggered(self):
        return self.expr.staggered

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

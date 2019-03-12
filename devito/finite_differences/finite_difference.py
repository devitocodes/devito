from functools import wraps

from sympy import S, finite_diff_weights

from devito.finite_differences import Differentiable
from devito.tools import Tag

__all__ = ['first_derivative', 'second_derivative', 'cross_derivative',
           'generic_derivative', 'left', 'right', 'centered', 'transpose',
           'generate_indices', 'form_side']

# Number of digits for FD coefficients to avoid roundup errors and non-deterministic
# code generation
_PRECISION = 9


class Transpose(Tag):
    """
    Utility class to change the sign of a derivative. This is only needed
    for odd order derivatives, which require a minus sign for the transpose.
    """
    pass


direct = Transpose('direct', 1)
transpose = Transpose('transpose', -1)


class Side(Tag):
    """
    Class encapsulating the side of the shift for derivatives.
    """

    def adjoint(self, matvec):
        if matvec == direct:
            return self
        else:
            if self == centered:
                return centered
            elif self == right:
                return left
            elif self == left:
                return right
            else:
                raise ValueError("Unsupported side value")


left = Side('left', -1)
right = Side('right', 1)
centered = Side('centered', 0)


def check_input(func):
    @wraps(func)
    def wrapper(expr, *args, **kwargs):
        if expr.is_Number:
            return S.Zero
        elif not isinstance(expr, Differentiable):
            raise ValueError("`%s` must be of type Differentiable (found `%s`)"
                             % (expr, type(expr)))
        else:
            return func(expr, *args, **kwargs)
    return wrapper


def check_symbolic(func):
    @wraps(func)
    def wrapper(expr, *args, **kwargs):
        if expr._uses_symbolic_coefficients:
            expr_dict = expr.as_coefficients_dict()
            if any(len(expr_dict) > 1 for item in expr_dict):
                raise NotImplementedError("Applying the chain rule to functions "
                                          "with symbolic coefficients is not currently "
                                          "supported")
        kwargs['symbolic'] = expr._uses_symbolic_coefficients
        return func(expr, *args, **kwargs)
    return wrapper


@check_input
@check_symbolic
def first_derivative(expr, dim, fd_order=None, side=centered, matvec=direct,
                     symbolic=False):
    """
    First-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the first-order derivative is produced.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int, optional
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil. Defaults to ``expr.space_order``
    side : Side, optional
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1). Defaults to ``centered``.
    matvec : Transpose, optional
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference. Defaults to ``direct``.

    Returns
    -------
    expr-like
        First-order derivative of ``expr``.

    Examples
    --------
    >>> from devito import Function, Grid, first_derivative, transpose
    >>> grid = Grid(shape=(4, 4))
    >>> x, _ = grid.dimensions
    >>> f = Function(name='f', grid=grid)
    >>> g = Function(name='g', grid=grid)
    >>> first_derivative(f*g, dim=x)
    -f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x

    This is also more easily obtainable via:

    >>> (f*g).dx.evaluate
    -f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x

    The adjoint mode

    >>> g = Function(name='g', grid=grid)
    >>> first_derivative(f*g, dim=x, matvec=transpose)
    f(x, y)*g(x, y)/h_x - f(x + h_x, y)*g(x + h_x, y)/h_x
    """

    side = side.adjoint(matvec)
    diff = dim.spacing
    adjoint_val = matvec.val
    order = fd_order or expr.space_order

    deriv = 0
    # Stencil positions for non-symmetric cross-derivatives with symmetric averaging
    ind = generate_indices(expr, dim, diff, order, side=side)[0]

    # Finite difference weights from Taylor approximation with this positions
    if symbolic:
        c = symbolic_weights(expr, 1, ind, dim)
    else:
        c = finite_diff_weights(1, ind, dim)[-1][-1]
    all_dims = tuple(set((dim,) + tuple([i for i in expr.indices if i.root == dim])))
    # Loop through positions
    for i in range(0, len(ind)):
        subs = dict([(d, ind[i].subs({dim: d})) for d in all_dims])
        deriv += expr.subs(subs) * c[i]

    return (adjoint_val*deriv).evalf(_PRECISION)


@check_input
@check_symbolic
def second_derivative(expr, dim, fd_order, stagger=None, **kwargs):
    """
    Second-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the derivative is produced.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    stagger : Side, optional
        Shift of the finite-difference approximation.

    Returns
    -------
    expr-like
        Second-order derivative of ``expr``.

    Examples
    --------
    >>> from devito import Function, Grid, second_derivative
    >>> grid = Grid(shape=(4, 4))
    >>> x, _ = grid.dimensions
    >>> f = Function(name='f', grid=grid, space_order=2)
    >>> g = Function(name='g', grid=grid, space_order=2)
    >>> second_derivative(f*g, dim=x, fd_order=2)
    -2.0*f(x, y)*g(x, y)/h_x**2 + f(x - h_x, y)*g(x - h_x, y)/h_x**2 +\
 f(x + h_x, y)*g(x + h_x, y)/h_x**2

    This is also more easily obtainable via:

    >>> (f*g).dx2.evaluate
    -2.0*f(x, y)*g(x, y)/h_x**2 + f(x - h_x, y)*g(x - h_x, y)/h_x**2 +\
 f(x + h_x, y)*g(x + h_x, y)/h_x**2
    """

    return generic_derivative(expr, dim, fd_order, 2, stagger=None, **kwargs)


@check_input
@check_symbolic
def cross_derivative(expr, dims, fd_order, deriv_order, stagger=None, **kwargs):
    """
    Arbitrary-order cross derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the cross derivative is produced.
    dims : tuple of Dimension
        Dimensions w.r.t. which to differentiate.
    fd_order : tuple of ints
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    deriv_order : tuple of ints
        Derivative order, e.g. 2 for a second-order derivative.
    stagger : tuple of Side, optional
        Shift of the finite-difference approximation.

    Returns
    -------
    expr-like
        Cross-derivative of ``expr``.

    Examples
    --------
    >>> from devito import Function, Grid, second_derivative
    >>> grid = Grid(shape=(4, 4))
    >>> x, y = grid.dimensions
    >>> f = Function(name='f', grid=grid, space_order=2)
    >>> g = Function(name='g', grid=grid, space_order=2)
    >>> cross_derivative(f*g, dims=(x, y), fd_order=(2, 2), deriv_order=(1, 1))
    -0.5*(-0.5*f(x - h_x, y - h_y)*g(x - h_x, y - h_y)/h_x +\
 0.5*f(x + h_x, y - h_y)*g(x + h_x, y - h_y)/h_x)/h_y +\
 0.5*(-0.5*f(x - h_x, y + h_y)*g(x - h_x, y + h_y)/h_x +\
 0.5*f(x + h_x, y + h_y)*g(x + h_x, y + h_y)/h_x)/h_y

    This is also more easily obtainable via:

    >>> (f*g).dxdy.evaluate
    -0.5*(-0.5*f(x - h_x, y - h_y)*g(x - h_x, y - h_y)/h_x +\
 0.5*f(x + h_x, y - h_y)*g(x + h_x, y - h_y)/h_x)/h_y +\
 0.5*(-0.5*f(x - h_x, y + h_y)*g(x - h_x, y + h_y)/h_x +\
 0.5*f(x + h_x, y + h_y)*g(x + h_x, y + h_y)/h_x)/h_y
    """

    stagger = stagger or [None]*len(dims)
    for d, fd, dim, s in zip(deriv_order, fd_order, dims, stagger):
        expr = generic_derivative(expr, dim=dim, fd_order=fd, deriv_order=d, stagger=s)

    return expr


@check_input
@check_symbolic
def generic_derivative(expr, dim, fd_order, deriv_order, stagger=None, symbolic=False,
                       matvec=direct):
    """
    Arbitrary-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the derivative is produced.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    deriv_order : int
        Derivative order, e.g. 2 for a second-order derivative.
    stagger : Side, optional
        Shift of the finite-difference approximation.

    Returns
    -------
    expr-like
        ``deriv-order`` derivative of ``expr``.
    """

    diff = dim.spacing
    adjoint_val = matvec.val**deriv_order
    indices, x0 = generate_indices(expr, dim, diff, fd_order, stagger=stagger)

    if symbolic:
        c = symbolic_weights(expr, deriv_order, indices, x0)
    else:
        c = finite_diff_weights(deriv_order, indices, x0)[-1][-1]

    deriv = 0
    all_dims = tuple(set((dim, ) +
                     tuple([i for i in expr.indices if i.root == dim])))
    for i in range(0, len(indices)):
        subs = dict((d, indices[i].subs({dim: d})) for d in all_dims)
        deriv += expr.subs(subs) * c[i]

    return (adjoint_val*deriv).evalf(_PRECISION)


def symbolic_weights(function, deriv_order, indices, dim):
    return [function._coeff_symbol(indices[j], deriv_order, function, dim)
            for j in range(0, len(indices))]


def generate_indices(func, dim, diff, order, stagger=None, side=None):

    # Check if called from first_derivative()
    if bool(side):
        if side == right:
            ind = [(dim+i*diff) for i in range(-int(order/2)+1-(order % 2),
                                               int((order+1)/2)+2-(order % 2))]
        elif side == left:
            ind = [(dim-i*diff) for i in range(-int(order/2)+1-(order % 2),
                                               int((order+1)/2)+2-(order % 2))]
        else:
            ind = [(dim+i*diff) for i in range(-int(order/2),
                                               int((order+1)/2)+1)]
        x0 = None
    else:
        if func.is_Staggered:
            if stagger == left:
                off = -.5
            elif stagger == right:
                off = .5
            else:
                off = 0
            ind = list(set([(dim + int(i+.5+off) * dim.spacing)
                            for i in range(-order//2, order//2)]))
            x0 = (dim + off*diff)
            if order < 2:
                ind = [dim + diff, dim] if stagger == right else [dim - diff, dim]

        else:
            ind = [(dim + i*dim.spacing) for i in range(-order//2, order//2 + 1)]
            x0 = dim
            if order < 2:
                ind = [dim, dim + diff]
    return ind, x0


def form_side(dimensions, function):
    side = dict()
    for (d, s) in zip(dimensions, function.staggered):
        if s == 0:
            side[d] = left
        elif s == 1:
            side[d] = right
        else:
            side[d] = centered
    return side

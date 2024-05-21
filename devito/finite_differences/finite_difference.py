from sympy import sympify

from .differentiable import EvalDerivative, DiffDerivative, Weights
from .tools import (left, right, generate_indices, centered, direct, transpose,
                    check_input, check_symbolic, fd_weights_registry)

__all__ = ['first_derivative', 'cross_derivative', 'generic_derivative',
           'left', 'right', 'centered', 'transpose', 'generate_indices']

# Number of digits for FD coefficients to avoid roundup errors and non-deterministic
# code generation
_PRECISION = 9


@check_input
@check_symbolic
def first_derivative(expr, dim, fd_order=None, side=centered, matvec=direct, x0=None,
                     coefficients='taylor', expand=True):
    """
    First-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the first-order derivative is produced.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int, optional, default=expr.space_order
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    side : Side, optional, default=centered
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1).
    matvec : Transpose, optional, default=direct
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference.
    x0 : dict, optional, default=None
        Origin of the finite-difference scheme as a map dim: origin_dim.
    coefficients : string, optional, default='taylor'
        Use taylor or custom coefficients (weights).
    expand : bool, optional, default=True
        If True, the derivative is fully expanded as a sum of products,
        otherwise an IndexSum is returned.

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

    Semantically, this is equivalent to

    >>> (f*g).dx
    Derivative(f(x, y)*g(x, y), x)

    The only difference is that in the latter case derivatives remain unevaluated.
    The expanded form is obtained via ``evaluate``

    >>> (f*g).dx.evaluate
    -f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x

    For the adjoint mode of the first derivative, pass ``matvec=transpose``

    >>> g = Function(name='g', grid=grid)
    >>> first_derivative(f*g, dim=x, matvec=transpose)
    -f(x, y)*g(x, y)/h_x + f(x - h_x, y)*g(x - h_x, y)/h_x

    This is also accessible via the .T shortcut

    >>> (f*g).dx.T.evaluate
    -f(x, y)*g(x, y)/h_x + f(x - h_x, y)*g(x - h_x, y)/h_x

    Finally the x0 argument allows to choose the origin of the finite-difference

    >>> first_derivative(f, dim=x, x0={x: x + x.spacing})
    -f(x + h_x, y)/h_x + f(x + 2*h_x, y)/h_x

    or specifying a specific location

    >>> first_derivative(f, dim=x, x0={x: 1})
    f(1, y)/h_x - f(1 - h_x, y)/h_x

    """
    fd_order = fd_order or expr.space_order
    deriv_order = 1

    # Enforce stable time coefficients
    if dim.is_Time and coefficients != 'symbolic':
        coefficients = 'taylor'

    return make_derivative(expr, dim, fd_order, deriv_order, side,
                           matvec, x0, coefficients, expand)


@check_input
@check_symbolic
def cross_derivative(expr, dims, fd_order, deriv_order, x0=None, **kwargs):
    """
    Arbitrary-order cross derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the cross derivative is produced.
    dims : tuple of Dimension
        Dimensions w.r.t. which to differentiate.
    fd_order : int, optional, default=expr.space_order
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    side : Side, optional, default=centered
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1).
    matvec : Transpose, optional, default=direct
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference.
    x0 : dict, optional, default=None
        Origin of the finite-difference scheme as a map dim: origin_dim.
    coefficients : string, optional, default='taylor'
        Use taylor or custom coefficients (weights).
    expand : bool, optional, default=True
        If True, the derivative is fully expanded as a sum of products,
        otherwise an IndexSum is returned.

    Returns
    -------
    expr-like
        Cross-derivative of ``expr``.

    Examples
    --------
    >>> from devito import Function, Grid
    >>> grid = Grid(shape=(4, 4))
    >>> x, y = grid.dimensions
    >>> f = Function(name='f', grid=grid, space_order=2)
    >>> g = Function(name='g', grid=grid, space_order=2)
    >>> cross_derivative(f*g, dims=(x, y), fd_order=(2, 2), deriv_order=(1, 1))
    (-1/h_y)*(-f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x) + \
(-f(x, y + h_y)*g(x, y + h_y)/h_x + f(x + h_x, y + h_y)*g(x + h_x, y + h_y)/h_x)/h_y

    Semantically, this is equivalent to

    >>> (f*g).dxdy
    Derivative(f(x, y)*g(x, y), x, y)

    The only difference is that in the latter case derivatives remain unevaluated.
    The expanded form is obtained via ``evaluate``

    >>> (f*g).dxdy.evaluate
    (-1/h_y)*(-f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x) + \
(-f(x, y + h_y)*g(x, y + h_y)/h_x + f(x + h_x, y + h_y)*g(x + h_x, y + h_y)/h_x)/h_y

    Finally the x0 argument allows to choose the origin of the finite-difference

    >>> cross_derivative(f*g, dims=(x, y), fd_order=(2, 2), deriv_order=(1, 1), \
    x0={x: x + x.spacing, y: y + y.spacing})
    (-1/h_y)*(-f(x + h_x, y + h_y)*g(x + h_x, y + h_y)/h_x + \
f(x + 2*h_x, y + h_y)*g(x + 2*h_x, y + h_y)/h_x) + \
(-f(x + h_x, y + 2*h_y)*g(x + h_x, y + 2*h_y)/h_x + \
f(x + 2*h_x, y + 2*h_y)*g(x + 2*h_x, y + 2*h_y)/h_x)/h_y
    """
    x0 = x0 or {}
    for d, fd, dim in zip(deriv_order, fd_order, dims):
        expr = generic_derivative(expr, dim=dim, fd_order=fd, deriv_order=d, x0=x0,
                                  **kwargs)

    return expr


@check_input
@check_symbolic
def generic_derivative(expr, dim, fd_order, deriv_order, matvec=direct, x0=None,
                       coefficients='taylor', expand=True):
    """
    Arbitrary-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the derivative is produced.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int, optional, default=expr.space_order
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    side : Side, optional, default=centered
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1).
    matvec : Transpose, optional, default=direct
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference.
    x0 : dict, optional, default=None
        Origin of the finite-difference scheme as a map dim: origin_dim.
    coefficients : string, optional, default='taylor'
        Use taylor or custom coefficients (weights).
    expand : bool, optional, default=True
        If True, the derivative is fully expanded as a sum of products,
        otherwise an IndexSum is returned.

    Returns
    -------
    expr-like
        ``deriv-order`` derivative of ``expr``.
    """
    side = None
    # First order derivative with 2nd order FD is strongly discouraged so taking
    # first order fd that is a lot better
    if deriv_order == 1 and fd_order == 2:
        fd_order = 1

    # Zeroth order derivative is just the expression itself if not shifted
    if deriv_order == 0 and not x0:
        return expr

    # Enforce stable time coefficients
    if dim.is_Time and coefficients != 'symbolic':
        coefficients = 'taylor'

    return make_derivative(expr, dim, fd_order, deriv_order, side,
                           matvec, x0, coefficients, expand)


def make_derivative(expr, dim, fd_order, deriv_order, side, matvec, x0, coefficients,
                    expand):
    # Always expand time derivatives to avoid issue with buffering and streaming.
    # Time derivative are almost always short stencils and won't benefit from
    # unexpansion in the rare case the derivative is not evaluated for time stepping.
    expand = True if dim.is_Time else expand

    # The stencil indices
    indices, x0 = generate_indices(expr, dim, fd_order, side=side, matvec=matvec,
                                   x0=x0)
    # Finite difference weights corresponding to the indices. Computed via the
    # `coefficients` method (`taylor` or `symbolic`)
    weights = fd_weights_registry[coefficients](expr, deriv_order, indices, x0)

    # Enforce fixed precision FD coefficients to avoid variations in results
    weights = [sympify(w).evalf(_PRECISION) for w in weights]

    # Transpose the FD, if necessary
    if matvec == transpose:
        weights = weights[::-1]
        indices = indices.transpose()

    # Shift index due to staggering, if any
    indices = indices.shift(-(expr.indices_ref[dim] - dim))

    # The user may wish to restrict expansion to selected derivatives
    if callable(expand):
        expand = expand(dim)

    if not expand and indices.expr is not None:
        weights = Weights(name='w', dimensions=indices.free_dim, initvalue=weights)

        # Inject the StencilDimension
        # E.g. `x + i*h_x` into `f(x)` s.t. `f(x + i*h_x)`
        expr = expr._subs(dim, indices.expr)

        # Re-evaluate any off-the-grid Functions potentially impacted by the FD
        try:
            expr = expr._evaluate(expand=False)
        except AttributeError:
            # Pure number
            pass

        deriv = DiffDerivative(expr*weights, {dim: indices.free_dim})
    else:
        terms = []
        for i, c in zip(indices, weights):
            # The FD term
            term = expr._subs(dim, i) * c

            # Re-evaluate any off-the-grid Functions potentially impacted by the FD
            try:
                term = term.evaluate
            except AttributeError:
                # Pure number
                pass

            terms.append(term)

        deriv = EvalDerivative(*terms, base=expr)

    return deriv

from sympy import sympify

from .differentiable import EvalDerivative, DiffDerivative, Weights
from .tools import (left, right, generate_indices, centered, direct, transpose,
                    check_input, fd_weights_registry, process_weights)

__all__ = ['first_derivative', 'cross_derivative', 'generic_derivative',
           'left', 'right', 'centered', 'transpose', 'generate_indices']

# Number of digits for FD coefficients to avoid roundup errors and non-deterministic
# code generation
_PRECISION = 9


@check_input
def cross_derivative(expr, dims, fd_order, deriv_order, x0=None, side=None, **kwargs):
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
                                  side=side, **kwargs)

    return expr


@check_input
def generic_derivative(expr, dim, fd_order, deriv_order, matvec=direct, x0=None,
                       coefficients='taylor', expand=True, weights=None, side=None):
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
    # First order derivative with 2nd order FD is strongly discouraged so taking
    # first order fd that is a lot better
    if deriv_order == 1 and fd_order == 2 and side is None:
        fd_order = 1

    # Zeroth order derivative is just the expression itself if not shifted
    if deriv_order == 0 and not x0:
        return expr

    # Enforce stable time coefficients
    if dim.is_Time:
        coefficients = 'taylor'
    else:
        coefficients = expr.coefficients

    return make_derivative(expr, dim, fd_order, deriv_order, side,
                           matvec, x0, coefficients, expand, weights)


# Backward compatibility
def first_derivative(expr, dim, fd_order, **kwargs):
    return generic_derivative(expr, dim, fd_order, 1, **kwargs)


def make_derivative(expr, dim, fd_order, deriv_order, side, matvec, x0, coefficients,
                    expand, weights=None):
    # Always expand time derivatives to avoid issue with buffering and streaming.
    # Time derivative are almost always short stencils and won't benefit from
    # unexpansion in the rare case the derivative is not evaluated for time stepping.
    expand = True if dim.is_Time else expand

    # The stencil indices
    nweights, wdim, scale = process_weights(weights, expr, dim)
    indices, x0 = generate_indices(expr, dim, fd_order, side=side, matvec=matvec,
                                   x0=x0, nweights=nweights)
    # Finite difference weights corresponding to the indices. Computed via the
    # `coefficients` method (`taylor` or `symbolic`)
    if weights is None:
        weights = fd_weights_registry[coefficients](expr, deriv_order, indices, x0)
    # Did fd_weights_registry return a new Function/Expression instead of a values?
    _, wdim, _ = process_weights(weights, expr, dim)
    if wdim is not None:
        weights = [weights._subs(wdim, i) for i in range(len(indices))]

    # Enforce fixed precision FD coefficients to avoid variations in results
    if scale:
        scale = dim.spacing**(-deriv_order)
    else:
        scale = 1
    weights = [sympify(scale * w).evalf(_PRECISION) for w in weights]

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

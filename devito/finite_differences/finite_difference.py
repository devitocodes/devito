from sympy import sympify

from .differentiable import EvalDerivative, IndexDerivative, Weights
from .tools import (numeric_weights, symbolic_weights, left, right,
                    generate_indices, centered, direct, transpose,
                    check_input, check_symbolic)

__all__ = ['first_derivative', 'cross_derivative', 'generic_derivative',
           'left', 'right', 'centered', 'transpose', 'generate_indices']

# Number of digits for FD coefficients to avoid roundup errors and non-deterministic
# code generation
_PRECISION = 9


@check_input
@check_symbolic
def first_derivative(expr, dim, fd_order=None, side=centered, matvec=direct, x0=None,
                     symbolic=False, expand=True):
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
        the resulting stencil. Defaults to `expr.space_order`.
    side : Side, optional
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1). Defaults to `centered`.
    matvec : Transpose, optional
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference. Defaults to `direct`.
    x0 : dict, optional
        Origin of the finite-difference scheme as a map dim: origin_dim.
    symbolic : bool, optional
        Use default or custom coefficients (weights). Defaults to False.
    expand : bool, optional
        If True, the derivative is fully expanded as a sum of products,
        otherwise an IndexSum is returned. Defaults to True.

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

    >>> first_derivative(f, dim=x, x0={x: 1})
    -f(1, y)/h_x + f(h_x + 1, y)/h_x
    """
    fd_order = fd_order or expr.space_order
    deriv_order = 1

    return make_derivative(expr, dim, fd_order, deriv_order, side,
                           matvec, x0, symbolic, expand)


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
    fd_order : tuple of ints
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    deriv_order : tuple of ints
        Derivative order, e.g. 2 for a second-order derivative.
    matvec : Transpose, optional
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference. Defaults to `direct`.
    x0 : dict, optional
        Origin of the finite-difference scheme as a map dim: origin_dim.
    symbolic : bool, optional
        Use default or custom coefficients (weights). Defaults to False.
    expand : bool, optional
        If True, the derivative is fully expanded as a sum of products,
        otherwise an IndexSum is returned. Defaults to True.

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
    x0={x: 1, y: 2})
    (-1/h_y)*(-f(1, 2)*g(1, 2)/h_x + f(h_x + 1, 2)*g(h_x + 1, 2)/h_x) + (-f(1, h_y + 2)*\
g(1, h_y + 2)/h_x + f(h_x + 1, h_y + 2)*g(h_x + 1, h_y + 2)/h_x)/h_y
    """
    x0 = x0 or {}
    for d, fd, dim in zip(deriv_order, fd_order, dims):
        expr = generic_derivative(expr, dim=dim, fd_order=fd, deriv_order=d, x0=x0,
                                  **kwargs)

    return expr


@check_input
@check_symbolic
def generic_derivative(expr, dim, fd_order, deriv_order, matvec=direct, x0=None,
                       symbolic=False, expand=True):
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
    matvec : Transpose, optional
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference. Defaults to `direct`.
    x0 : dict, optional
        Origin of the finite-difference scheme as a map dim: origin_dim.
    symbolic : bool, optional
        Use default or custom coefficients (weights). Defaults to False.
    expand : bool, optional
        If True, the derivative is fully expanded as a sum of products,
        otherwise an IndexSum is returned. Defaults to True.

    Returns
    -------
    expr-like
        ``deriv-order`` derivative of ``expr``.
    """
    side = None
    # First order derivative with 2nd order FD is highly non-recommended so taking
    # first order fd that is a lot better
    if deriv_order == 1 and fd_order == 2 and not symbolic:
        fd_order = 1

    return make_derivative(expr, dim, fd_order, deriv_order, side,
                           matvec, x0, symbolic, expand)


def make_derivative(expr, dim, fd_order, deriv_order, side, matvec, x0, symbolic, expand):
    # The stencil indices
    indices, x0 = generate_indices(expr, dim, fd_order, side=side, matvec=matvec, x0=x0)

    # Finite difference weights from Taylor approximation given these positions
    if symbolic:
        weights = symbolic_weights(expr, deriv_order, indices, x0)
    else:
        weights = numeric_weights(deriv_order, indices, x0)

    # Enforce fixed precision FD coefficients to avoid variations in results
    weights = [sympify(w).evalf(_PRECISION) for w in weights]

    # Transpose the FD, if necessary
    if matvec:
        indices = indices.scale(matvec.val)

    # Shift index due to staggering, if any
    indices = indices.shift(-(expr.indices_ref[dim] - dim))

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

        deriv = IndexDerivative(expr*weights, {dim: indices.free_dim})
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

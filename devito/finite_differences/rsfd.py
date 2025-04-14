from functools import wraps

from devito.types.dimension import StencilDimension
from .differentiable import Weights, DiffDerivative
from .tools import generate_indices, fd_weights_registry

__all__ = ['drot', 'd45']

smapper = {1: (1, 1, 1), 2: (1, 1, -1), 3: (1, -1, 1), 4: (1, -1, -1)}


def shift(sign, x0):
    if x0 == 0:
        return 0 if sign > 0 else -1
    else:
        return 0 if sign > 0 else 1


def drot(expr, dim, dir=1, x0=None):
    """
    Finite difference approximation of the derivative along dir
    of a Function `f` at point `x0`.

    Rotated finite differences based on:
    https://www.sciencedirect.com/science/article/pii/S0165212599000232
    The rotated axes (the four diagonals of a cube) are:
      d1 = dx/dr x + dz/dl y + dy/dl z
      d2 = dx/dl x + dz/dl y - dy/dr z
      d3 = dx/dr x - dz/dl y + dy/dl z
      d4 = dx/dl x - dz/dl y - dy/dr z
    Or in 2d (the two diagonals of a square):
      d1 = dx/dr x + dy/dl y
      d2 = dx/dl x - dy/dr y
    """
    # 2D case
    ndim = expr.grid.dim
    if dir > 2 and ndim == 2:
        return 0

    # RSFD scaling
    s = 2**(expr.grid.dim - 1)

    # Center point and indices
    indices, start = generate_indices(expr, dim, expr.space_order, x0=x0)

    # a-dimensional center for FD coefficients.
    adim_start = x0.get(dim, expr.indices_ref[dim]).subs({dim: 0, dim.spacing: 1})
    mid = expr.indices_ref[dim].subs({dim: 0, dim.spacing: 1})

    # a-dimensional indices
    adim_indices = sorted([i.subs({dim: 0, dim.spacing: 1}) for i in indices])

    # FD coeffs
    # Dispersion reduction weights currently not working as the lsqr
    # system needs to be setup for the whole stencil
    coeffs = fd_weights_registry['taylor'](expr, 1, adim_indices, adim_start)
    i = StencilDimension('i', adim_indices[0]-mid, adim_indices[-1]-mid)
    w0 = Weights(name='w0', dimensions=i, initvalue=coeffs)

    # Skip y if 2D
    signs = smapper[dir][::(1 if ndim == 3 else 2)]

    # Direction substitutions
    dim_mapper = {}
    for (di, d) in enumerate(expr.grid.dimensions):
        s0 = 0 if mid == adim_start else shift(signs[di], mid)*d.spacing
        dim_mapper[d] = d + signs[di]*i*d.spacing - s0

    # Create IndexDerivative
    ui = expr.subs(dim_mapper)

    deriv = DiffDerivative(w0*ui/(s*dim.spacing), {d: i for d in expr.grid.dimensions})

    return deriv


grid_node = lambda grid: {d: d for d in grid.dimensions}
all_staggered = lambda grid: {d: d + d.spacing/2 for d in grid.dimensions}


def check_staggering(func):
    """
    Because of the very specific structure of the 45 degree stencil
    only two cases can happen:

    1. No staggering. In this case the stencil is center on the node where
       the Function/Expr is defined and the diagonal is well defined.
    2. Full staggering. The center node must be either NODE or grid.dimension so that
       the diagonal align with the staggering of the expression. I.e a NODE center point
       for a `grid.dimension` staggered expression
       or a `grid.dimension` center point for a NODE staggered expression.

    Therefore acceptable combinations are:
    - NODE center point and NODE staggering
    - grid.dimension center point and grid.dimension staggering
    - NODE center point and grid.dimension staggering
    - grid.dimension center point and NODE staggering
    """
    @wraps(func)
    def wrapper(expr, dim, x0=None, expand=True):
        grid = expr.grid
        x0 = {k: v for k, v in x0.items() if k.is_Space}
        cond = x0 == {} or x0 == all_staggered(grid) or x0 == grid_node(grid)
        if cond:
            return func(expr, dim, x0=x0, expand=expand)
        else:
            raise ValueError('Invalid staggering or x0 for rotated finite differences')
    return wrapper


@check_staggering
def d45(expr, dim, x0=None, expand=True):
    """
    Rotated staggered grid finite-differences (RSFD) discretization
    of the derivative of `expr` along `dim` at point `x0`.

    Parameters
    ----------
    expr : expr-like
        Expression for which the derivative is produced.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    x0 : dict, optional
        Origin of the finite-difference. Defaults to 0 for all dimensions.
    expand : bool, optional
        Expand the expression. Defaults to True.
    """
    # Make sure the grid supports RSFD
    if expr.grid.dim not in [2, 3]:
        raise ValueError('Rotated staggered grid finite-differences (RSFD)'
                         ' only supported in 2D and 3D')

    # Diagonals weights
    w = dir_weights[(dim.name, expr.grid.dim)]

    # RSFD
    rsfd = (w[0] * drot(expr, dim, x0=x0, dir=1) +
            w[1] * drot(expr, dim, x0=x0, dir=2) +
            w[2] * drot(expr, dim, x0=x0, dir=3) +
            w[3] * drot(expr, dim, x0=x0, dir=4))

    # Evaluate
    return rsfd._evaluate(expand=expand)


# How to sum d1, d2, d3, d4 depending on the dimension
dir_weights = {('x', 2): (1, 1, 1, 1), ('x', 3): (1, 1, 1, 1),
               ('y', 2): (1, -1, 1, -1), ('y', 3): (1, 1, -1, -1),
               ('z', 2): (1, -1, 1, -1), ('z', 3): (1, -1, 1, -1)}

from sympy import finite_diff_weights

from devito.types.dimension import StencilDimension
from .differentiable import Weights, DiffDerivative
from .tools import generate_indices_staggered
from .elementary import sqrt

__all__ = ['drot', 'dxrot', 'dyrot', 'dzrot']

smapper = {1: (1, 1, 1), 2: (1, 1, -1), 3: (1, -1, 1), 4: (1, -1, -1)}


def shift(sign, x0):
    if x0 == 0:
        return 0 if sign > 0 else -1
    else:
        return 0 if sign > 0 else 1


def drot(expr, dim, dir=1, x0=None):
    """
    Finite difference approximation of the derivative along d1
    of a Function `f` at point `x0`.

    Rotated finite differences based on:
    https://www.sciencedirect.com/science/article/pii/S0165212599000232
    The rotated axis (the four diagonal of a cube) are:
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

    # Spacing along diagonal
    r = sqrt(sum(d.spacing**2 for d in expr.grid.dimensions))

    # Center point
    start, indices = generate_indices_staggered(expr, dim, expr.space_order, x0=x0)
    adim_start = start.subs({dim: 0, dim.spacing: 1})
    adim_indices = [i.subs({dim: 0, dim.spacing: 1}) for i in indices]
    coeffs = finite_diff_weights(1, adim_indices, adim_start)[-1][-1]

    i = StencilDimension('i', adim_indices[0]-adim_start, adim_indices[-1]-adim_start)

    # FD weights
    w0 = Weights(name='w0', dimensions=i, initvalue=coeffs)

    # Skip y if 2D
    signs = smapper[dir][::(1 if ndim == 3 else 2)]

    # Direction substitutions
    dim_mapper = {d: d + signs[di]*i*d.spacing - shift(signs[di], adim_start)*d.spacing
                  for (di, d) in enumerate(expr.grid.dimensions)}

    # Create IndexDerivative
    ui = expr.subs(dim_mapper)

    deriv = DiffDerivative(w0*ui, {d: i for d in expr.grid.dimensions})

    return deriv/r


def dxrot(expr, x0=None, expand=True):
    x = expr.grid.dimensions[0]
    r = sqrt(sum(d.spacing**2 for d in expr.grid.dimensions))
    s = 2**(expr.grid.dim - 1)
    dxrsfd = (drot(expr, x, x0=x0, dir=1) + drot(expr, x, x0=x0, dir=2) +
              drot(expr, x, x0=x0, dir=3) + drot(expr, x, x0=x0, dir=4))
    dx45 = r / (s * x.spacing) * dxrsfd
    if expand:
        return dx45.evaluate
    else:
        return dx45


def dyrot(expr, x0=None, expand=True):
    y = expr.grid.dimensions[1]
    r = sqrt(sum(d.spacing**2 for d in expr.grid.dimensions))
    s = 2**(expr.grid.dim - 1)
    dyrsfd = (drot(expr, y, x0=x0, dir=1) + drot(expr, y, x0=x0, dir=2) -
              drot(expr, y, x0=x0, dir=3) - drot(expr, y, x0=x0, dir=4))
    dy45 = r / (s * y.spacing) * dyrsfd
    if expand:
        return dy45.evaluate
    else:
        return dy45


def dzrot(expr, x0=None, expand=True):
    z = expr.grid.dimensions[-1]
    r = sqrt(sum(d.spacing**2 for d in expr.grid.dimensions))
    s = 2**(expr.grid.dim - 1)
    dzrsfd = (drot(expr, z, x0=x0, dir=1) - drot(expr, z, x0=x0, dir=2) +
              drot(expr, z, x0=x0, dir=3) - drot(expr, z, x0=x0, dir=4))
    dz45 = r / (s * z.spacing) * dzrsfd
    if expand:
        return dz45.evaluate
    else:
        return dz45


difrot = {2: {'dx': dxrot, 'dy': dzrot}, 3: {'dx': dxrot, 'dy': dyrot, 'dz': dzrot}}

"""
The free surface of the seismic models, written as equations on the fields.

A model with a free surface has `s` rows above the surface, which is row `s`.
After each update, `mirror` copies the field into these rows. For an odd image,
`u(z) = -u(2s - z)`, and the field is zero on the surface. For an even image,
`u(z) = u(2s - z)`. A field staggered along the vertical uses `2s - 1 - z`
instead. On a CurvilinearGrid the rows follow the surface, so the same equations
mirror along the grid lines.

`fold` is the adjoint of `mirror`. Under MPI, each rank along the vertical needs
at least `2s + 1` rows, so that the rows the mirror reads are in its domain or
its halo.
"""

import sympy

from devito import Eq, Inc, solve

__all__ = ['fold', 'mirror', 'solve_update', 'update_domain']


def mirror(model, field, odd=True, zero=None):
    """
    Return the free-surface equations of a field after its update.

    Parameters
    ----------
    model : SeismicModel
        The model, with a free surface or not.
    field : TimeFunction
        The updated field, e.g. `u.forward`.
    odd : bool, optional, default=True
        Whether the image is odd or even.
    zero : bool, optional
        Whether to set the field to zero on the surface row, if it has a point
        there. Defaults to True for an odd image only.

    Returns
    -------
    list of Eq
        The equations that mirror the field and, if `zero`, set it to zero on
        the surface. Empty without a free surface.
    """
    rows = model.free_surface_rows
    if rows is None:
        return []

    s, shift = model.mirror_rows, _shift(model, field)
    z = model.grid.dimensions[-1]
    zero = odd if zero is None else zero
    eqs = [Eq(field, 0, subdomain=rows.surface)] if zero and not shift else []
    sign = -1 if odd else 1
    eqs.append(Eq(field, sign*field.subs({z: 2*s - shift - z}), subdomain=rows.above))
    return eqs


def fold(model, field, coefficient, odd=True, zero=None):
    """
    Return the adjoint of `mirror`, to apply after an adjoint update.

    It is the adjoint in the inner product weighted by the Jacobian `J`, which
    is 1 on a Grid. It adds the mirror rows, scaled, onto the rows they mirror,
    `v(z) += -c/J (J v/c)(2s - z)` for an odd image, then sets them to zero.

    Parameters
    ----------
    model : SeismicModel
        The model, with a free surface or not.
    field : TimeFunction
        The updated adjoint field, e.g. `v.backward`.
    coefficient : expr-like
        The factor `c` of the spatial operator in the update (see
        `solve_update`). Use 0 for an update without a spatial operator, e.g. of
        a memory variable: its mirror rows are then only set to zero.
    odd : bool, optional, default=True
        Whether the forward `mirror` uses an odd image.
    zero : bool, optional
        Whether the forward `mirror` sets the surface row to zero. Defaults to
        True for an odd image only.

    Returns
    -------
    list of Eq
        The equations of the fold. Empty without a free surface.
    """
    rows = model.free_surface_rows
    if rows is None:
        return []

    s, shift = model.mirror_rows, _shift(model, field)
    grid = model.grid
    z = grid.dimensions[-1]

    eqs = []
    if coefficient != 0:
        location = {d: field.indices_ref[d] for d in grid.dimensions}
        jacobian = grid.jacobian_at(location)
        inverse = grid.inverse_jacobian_at(location)
        folded = (jacobian*field/coefficient).subs({z: 2*s - shift - z})
        sign = -1 if odd else 1
        mirrored = rows.mirrored_staggered if shift else rows.mirrored
        eqs.append(Inc(field, sign*coefficient*inverse*folded, subdomain=mirrored))
    eqs.append(Eq(field, 0, subdomain=rows.above))
    if (odd if zero is None else zero) and not shift:
        eqs.append(Eq(field, 0, subdomain=rows.surface))
    return eqs


def solve_update(pde, spatial, target):
    """
    Solve `pde - spatial = 0` for `target`.

    Return the update, and the factor of `spatial` in it, which `fold` needs.

    Parameters
    ----------
    pde : expr-like
        The time derivatives and the sources, e.g. `m*u.dt2 + damp*u.dt`.
    spatial : expr-like
        The spatial operator, e.g. `u.laplace`.
    target : TimeFunction
        The updated field, e.g. `u.forward`.

    Returns
    -------
    tuple
        The update, and the factor of `spatial` in it, e.g.
        `dt**2/(m + damp*dt)` for the acoustic wave equation backward in time.
    """
    operator = sympy.Dummy('spatial')
    update = solve(pde - operator, target)
    factor = sympy.expand_mul(update).coeff(operator)
    return update.xreplace({operator: spatial}), factor


def update_domain(model, forward=True):
    """
    Return the SubDomain of the update of a field.

    Parameters
    ----------
    model : SeismicModel
        The model, with a free surface or not.
    forward : bool, optional, default=True
        Whether the update is forward in time, or adjoint.

    Returns
    -------
    SubDomain or None
        For a forward update with a free surface, the rows below the mirror
        rows. Otherwise None, which is the whole grid: an adjoint update also
        covers the mirror rows, and `fold` then folds them back.
    """
    if model.free_surface_rows is not None and forward:
        return model.updated_rows
    return None


def _shift(model, field):
    """Return 1 for a field staggered along the vertical Dimension, else 0."""
    z = model.grid.dimensions[-1]
    return int(field.indices_ref[z] != z)

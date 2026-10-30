"""
The free surface of the seismic models, written as equations of the fields.

A model with a free surface, `fs=True`, has `s = model.mirror_rows` rows above
the surface row `s`. After each update, the free surface mirrors a field into
the rows above: an odd image through the surface, `u(z) = -u(2 s - z)`, zero
on the surface, as for a pressure that vanishes there, or an even one,
`u(z) = u(2 s - z)`, which may be zeroed on the surface too. A field
staggered along the vertical Dimension has no point on the surface, and its
image is `u(z) = -u(2 s - 1 - z)` or `u(2 s - 1 - z)`. On a CurvilinearGrid
the rows follow the surface, so the same equations mirror the field along
the grid lines.

The adjoint of an update followed by the mirror is the fold of the mirror
rows onto the rows they mirror, `fold`, followed by the adjoint update, which
then covers the mirror rows too (see `update_domain`).
"""

import sympy

from devito import Eq, Inc, solve
from devito.mpi import MPI

__all__ = ['fold', 'mirror', 'solve_update', 'update_domain']


def mirror(model, field, odd=True, zero=None):
    """
    The free surface of a field after its update.

    Parameters
    ----------
    model : SeismicModel
        The model, with a free surface or not.
    field : TimeFunction
        The updated field, e.g. `u.forward`.
    odd : bool, optional, default=True
        An odd image or an even one.
    zero : bool, optional
        Whether the field is zero on the surface row, if it has a point there,
        by default for an odd image only.

    Returns
    -------
    list of Eq
        The field mirrored into the rows above the surface, and zero on the
        surface row if `zero`, or no equations without a free surface.
    """
    rows = model.free_surface_rows
    if rows is None:
        return []

    s, shift = model.mirror_rows, _shift(model, field)
    z = model.grid.dimensions[-1]
    _check_reach(model, field, (0, s), shift)
    zero = odd if zero is None else zero
    eqs = [Eq(field, 0, subdomain=rows.surface)] if zero and not shift else []
    sign = -1 if odd else 1
    eqs.append(Eq(field, sign*field.subs({z: 2*s - shift - z}), subdomain=rows.above))
    return eqs


def fold(model, field, coefficient, odd=True, zero=None):
    """
    The transpose of `mirror`, after an adjoint update.

    It is the transpose in the inner product weighted by the Jacobian `J` of
    the grid, 1 on a Grid. The adjoint update holds, in the mirror rows, the
    spatial operator times `coefficient`; these rows are folded onto the rows
    below the surface that they mirror, `v(z) += -c/J (J v/c)(2 s - z)` for an
    odd image, then zeroed, with the surface row if the mirror zeroes it.

    Parameters
    ----------
    model : SeismicModel
        The model, with a free surface or not.
    field : TimeFunction
        The updated adjoint field, e.g. `v.backward`.
    coefficient : expr-like
        The factor of the spatial operator in the update, e.g. `dt**2/(m +
        damp*dt)` for the acoustic one (see `solve_update`), or 0 for an update
        without one, e.g. of a memory variable, whose mirror rows are only
        zeroed.
    odd : bool, optional, default=True
        The image of the forward `mirror`.
    zero : bool, optional
        Whether the forward `mirror` zeroes the surface row, by default for an
        odd image only.

    Returns
    -------
    list of Eq
        The fold, or no equations without a free surface.
    """
    rows = model.free_surface_rows
    if rows is None:
        return []

    s, shift = model.mirror_rows, _shift(model, field)
    grid = model.grid
    z = grid.dimensions[-1]
    _check_reach(model, field, (s + 1 - shift, 2*s + 1 - shift), shift)

    eqs = []
    if coefficient != 0:
        # The Jacobian where the values of the field are
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
    The update of `target` solving `pde - spatial = 0`, and the factor of the
    spatial operator in it, which `fold` needs.

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
        The update, and the factor of `spatial` in it, e.g. `dt**2/(m +
        damp*dt)` for the acoustic wave equation backward in time.
    """
    # The update is linear in the spatial operator: its factor is the
    # coefficient of a symbol solved for in its place
    operator = sympy.Dummy('spatial')
    update = solve(pde - operator, target)
    factor = sympy.expand_mul(update).coeff(operator)
    return update.xreplace({operator: spatial}), factor


def update_domain(model, forward=True):
    """
    The SubDomain of the update of a field.

    Parameters
    ----------
    model : SeismicModel
        The model, with a free surface or not.
    forward : bool, optional, default=True
        Whether the update is forward in time, or an adjoint one.

    Returns
    -------
    SubDomain or None
        With a free surface, the rows below the mirror rows for a forward
        update, else None, the whole grid: an adjoint update covers the
        mirror rows too, which `fold` then folds back.
    """
    if model.free_surface_rows is not None and forward:
        return model.physical_domain
    return None


def _shift(model, field):
    """1 for a field staggered along the vertical Dimension, else 0."""
    z = model.grid.dimensions[-1]
    return int(field.indices_ref[z] != z)


def _check_reach(model, field, written, shift):
    """
    Check that each rank writing the rows `written`, a `(start, stop)` range,
    of the mirror or the fold of `field` owns the rows they read,
    `2 s - shift - z`, or holds them in its halo.

    Raises
    ------
    ValueError
        On every rank, if some rank would read past its halo, e.g. with the
        vertical Dimension decomposed into ranks of fewer rows than the
        mirror reaches.
    """
    grid = model.grid
    if not grid.distributor.is_parallel:
        return

    s = model.mirror_rows
    owned = grid.distributor.glb_numb[-1]
    left, right = field.function.halo[-1]
    reached = True
    if len(owned):
        first, last = owned[0], owned[-1]
        rows = range(max(written[0], first), min(written[1], last + 1))
        reached = all(first - left <= 2*s - shift - r <= last + right for r in rows)

    if not grid.comm.allreduce(reached, op=MPI.LAND):
        raise ValueError(f"The free surface of {field} reads rows {s} rows across the "
                         "surface, which some rank holds neither in its domain nor in "
                         "its halo: decompose the vertical Dimension into ranks of at "
                         f"least {2*s + 1} rows")

"""
Reading a Function defined on a SubGrid at its parent Grid's points.

A SubGrid point `i` sits at parent coordinate `i*factor`, so the parent point
`x` sits at SubGrid coordinate `x*q/p` for a `factor = p/q`:

    base = (x*q) // p        x0 = ((x*q) % p) / p

`x0` depends on `x` only through `x % p`, so there are `p` *phases*, each a
zeroth-order derivative at a fixed `x0` -- the relocation a staggered Function
gets. Only the phase is chosen at run time.
"""

from collections import namedtuple
from itertools import product

import sympy

from devito.finite_differences.differentiable import (
    DiffDerivative, EvalDerivative, Mul, Weights, expands
)
from devito.finite_differences.tools import (
    generate_indices, kaiser_b, numeric_weights, sinc_weights
)
from devito.types.dimension import StencilDimension

# `p` sizes the weight table. Checked at the read, not at SubGrid creation: a
# large integer factor is still a fine strided write, which builds no table
_MAX_PHASES = 64

__all__ = ['is_subgrid_read', 'subgrid_interp']


class SubGridStencil(namedtuple('SubGridStencil',
                                'dim parent_dim weights index x0')):

    """
    How one Dimension of a SubGrid read is relocated.

    `index` is the rank-local SubGrid index the taps are relative to. `x0`
    picks a set of `weights` at run time, or None if there is only one.
    """


def is_subgrid_read(func, target, domain=None):
    """
    Whether reading `func` at `target`'s location is a SubGrid relocation.

    False if `target` is on the same SubGrid, shares none of its Dimensions,
    or is the iteration `domain`, which makes the read a plain scatter.
    """
    grid = func.function.grid
    if grid is None or not grid.is_SubGrid:
        return False

    if target.function.grid is grid:
        return False

    if domain is grid:
        return False

    return any(d in target.dimensions for d in grid.parent.dimensions)


def subgrid_interp(func, target, expand=True):
    """
    `func`, defined on a SubGrid, read at `target`'s points: a weighted sum
    over the taps of every relocated Dimension.

    With `expand` the taps are spelled out; otherwise each Dimension keeps a
    `DiffDerivative`, lowered to a loop rather than `interp_order**ndim`
    terms. `expand` may also be a per-Dimension predicate.
    """
    f = func.function
    grid = f.grid

    stencils = []
    for dim, parent_dim, factor in zip(grid.dimensions,
                                       grid.parent.dimensions,
                                       grid.rational_factor, strict=True):
        if dim is parent_dim or parent_dim not in target.dimensions:
            # Factor of 1, or an axis `target` does not iterate
            continue

        stencils.append(_stencil(f, dim, parent_dim, factor, target))

    if not stencils:
        return func

    # On the relocated Dimensions only: an unrelated axis must not force this
    # read to expand
    if not expands(expand, [i.parent_dim for i in stencils]):
        return _as_derivatives(func, stencils)

    return _as_taps(func, stencils)


def _stencil(f, dim, parent_dim, factor, target):
    """
    How `f` is read along `dim` at `target`'s points: the taps, their weights,
    and where they are centred.
    """
    # Staggering is just a different evaluation point
    shift = _cell_offset(target, parent_dim)

    weights = _phase_weights(f, dim, factor, shift)

    # `IntDiv` and `Mod` are not affine, so they take the global index;
    # `ioffset` brings the result back to rank-local. Both are 0 without MPI
    parent_ioffset, ioffset = f.grid.ioffsets[parent_dim]
    index = parent_dim + parent_ioffset

    return SubGridStencil(
        dim, parent_dim, weights,
        dim.subsampled_index(index, shift=shift) - ioffset,
        sympy.Mod(index, weights.nsets) if weights.nsets > 1 else None
    )


def _as_taps(func, stencils):
    """The read with every tap spelled out, as `finite_difference` expands one."""
    terms = []
    for taps in product(*[i.weights.offsets for i in stencils]):
        weight = sympy.S.One
        mapper = {}
        for stencil, tap in zip(stencils, taps, strict=True):
            w = stencil.weights
            weight = weight*w[w.index(tap - w.offsets[0], stencil.x0)]
            mapper[stencil.dim] = stencil.index + tap
        terms.append(Mul(weight, func.subs(mapper)))

    # As a staggered relocation comes back: evaluated, hence not relocated
    # again, and carrying the expression it reads
    return EvalDerivative(*terms, base=func)


def _as_derivatives(func, stencils):
    """The read as a nest of `DiffDerivative`s, one per relocated Dimension."""
    expr = func
    for stencil in stencils:
        w = stencil.weights
        free_dim = w.dimension

        # No shift on `free_dim`: `lower_index_derivatives` translates the
        # loop to 0-based and compensates in the base
        expr = DiffDerivative(
            expr.subs(stencil.dim, stencil.index + free_dim) *
            w[w.index(free_dim, stencil.x0)],
            {stencil.parent_dim: free_dim}, deriv_order=0
        )

    return expr


def _cell_offset(target, dim):
    """
    Where `target` is evaluated along `dim`, as a fraction of a cell: 0 for an
    ordinary Function, 1/2 for one staggered along `dim`.
    """
    # As `generate_indices` locates its own evaluation point
    offset = ((target.indices_ref[dim] - dim)/dim.spacing).subs({dim: 0})

    return sympy.sympify(offset).simplify()


def _phase_weights(f, dim, factor, shift=0):
    """
    The `Weights` for reading `f` at its parent's points along `dim`.

    One set per phase, `f.interp_order` wide, laid out phase-major. `shift`
    is where the read sits within a parent cell.
    """
    p, q = factor.numerator, factor.denominator

    # The read inverts the factor, so the period is `p`
    nphases = p

    if nphases > _MAX_PHASES:
        raise ValueError(
            f"Reading `{f.name}` at its parent's points along `{dim.name}` "
            f"needs {nphases} phases, more than the {_MAX_PHASES} supported, "
            f"each with its own set of weights. Use a factor that is a ratio "
            f"of small integers, or write the SubGrid rather than read it"
        )

    # Row `k` is read at parent index `x = k (mod p)`, i.e. SubGrid
    # coordinate `(b*q*k + a*q)/(b*p)` for `shift = a/b`; `x0` is its
    # fractional part
    a, b = sympy.Rational(shift).p, sympy.Rational(shift).q

    rows = []
    for phase in range(nphases):
        x0 = sympy.Rational((b*q*phase + a*q) % (b*p), b*p)
        rows.append(weights_at(f.grid.interpolation, f, dim, x0))

    # One tap layout for every phase, zero-filled where a phase does not
    # reach: a rectangular table makes the run-time row a `phase*ntaps` offset
    offsets = sorted(set().union(*[set(i) for i in rows]))
    values = [sympy.Float(i.get(o, 0.)) for i in rows for o in offsets]

    # Named after `dim` rather than numbered as `make_stencil_dimension`
    # numbers within one expression: two SubGrids in one expression are
    # contracted one inside the other and each needs its own
    free_dim = StencilDimension(f'i{dim.name}', offsets[0], offsets[-1])

    return Weights(name='w', dimensions=free_dim, initvalue=values,
                   dtype=f.dtype)


def _fd_weights_at(f, dim, x0):
    """
    `{offset -> weight}` for one phase: the zeroth-order FD stencil relocating
    `f` along `dim` to `x0`, the same one a staggered Function gets.

    One level under `f.diff(dim, deriv_order=0, x0=...)`, which would build a
    stencil per phase only for `_phase_weights` to take it apart again.
    """
    if not x0:
        # On a point of `f`: the FD scheme would return a centred stencil
        # weighted `[0, ..., 1, ..., 0]`
        return {0: sympy.S.One}

    indices, at = _indices_at(f, dim, x0)
    weights = numeric_weights(f, 0, indices, at)

    return dict(zip(indices.free_dim.range, weights, strict=True))


def _indices_at(f, dim, x0):
    """The FD scheme's points for relocating `f` along `dim` to `x0`."""
    return generate_indices(f, dim, f.interp_order,
                            x0={dim: dim + x0*dim.spacing})


def _sinc_weights_at(f, dim, x0):
    """
    `{offset -> weight}` for one phase of Kaiser windowed sinc, after Hicks
    (2002). Normalised: the truncated window sums to ~0.4% off unity, which
    would show up as a constant bias.
    """
    if not x0:
        # On a point of `f`, as for the FD weights
        return {0: sympy.S.One}

    # The taps are the FD scheme's; only the weights differ
    indices, _ = _indices_at(f, dim, x0)
    offsets = tuple(indices.free_dim.range)

    # `kaiser_b` is keyed by radius, as `SincInterpolator` indexes it
    r = f.interp_order//2
    w = sinc_weights([o - x0 for o in offsets], r, kaiser_b[r],
                     normalize=True)

    return dict(zip(offsets, w, strict=True))


# Per scheme: how to build one phase's weights, and the radii it supports.
# Shaped like `fd_weights_registry`
interp_schemes_registry = {
    'fd': (_fd_weights_at, None),
    'sinc': (_sinc_weights_at, frozenset(kaiser_b)),
}


def weights_at(interpolation, f, dim, x0):
    """One phase's `{offset -> weight}`, from the named scheme."""
    builder, _ = interp_schemes_registry[interpolation]

    return builder(f, dim, x0)


def supported_radii(interpolation):
    """
    The stencil radii the named scheme supports, or None if it supports any.
    """
    _, radii = interp_schemes_registry[interpolation]

    return radii

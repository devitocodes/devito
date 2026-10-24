"""
Reading a Function defined on a SubGrid at its parent Grid's points.

A SubGrid point `i` sits at parent coordinate `i*factor`, so conversely the
parent point `x` sits at SubGrid coordinate `x/factor`. With `factor = p/q`
that is `x*q/p`, hence

    base = (x*q) // p        frac = ((x*q) % p) / p

`base` is an ordinary integer-arithmetic index, and `frac` -- the position
within the cell, which is what the interpolation weights depend on -- varies
with `x` only through `x % p`. So there are `p` phases, and within a phase the
shift is a *constant*.

That is what lets the weights come from the ordinary FD path: each phase is a
zeroth-order derivative at a compile-time `x0`, exactly as a staggered Function
is relocated, and `interp_order` sets the stencil width. Only the *selection*
between phases is left to run time, as an index into a small table.
"""

from hashlib import sha1
from itertools import product

import sympy

from devito.finite_differences.differentiable import Add, Mul
from devito.finite_differences.tools import kaiser_b, numeric_weights, sinc_weights
from devito.symbolics import IntDiv
from devito.types import Array, CustomDimension

__all__ = ['is_subgrid_read', 'subgrid_interp']


def is_subgrid_read(func, target, domain=None):
    """
    Whether reading `func` at `target`'s location is a SubGrid relocation.

    True when `func` lives on a SubGrid and `target` does not live on that same
    SubGrid. The exception matters: a Function read at a shift of its own
    Dimensions, as a recurrence does, is not a relocation.

    Only the Dimensions `target` actually iterates can be relocated, so a
    SubGrid sharing none of them is not a relocation either.

    `domain` is the Eq's iteration domain, if given. Iterating the SubGrid
    itself means only its own points are visited, so there is nothing to
    relocate and the read is a plain scatter -- see `SubGrid`.
    """
    grid = func.function.grid
    if grid is None or not grid.is_SubGrid:
        return False

    if target.function.grid is grid:
        return False

    if domain is grid:
        return False

    return any(d in target.dimensions for d in grid.parent.dimensions)


def subgrid_interp(func, target):
    """
    `func`, defined on a SubGrid, read at `target`'s points: a weighted sum
    over the taps of every relocated Dimension.

    Returns the stencil as an ordinary expression, so that everything
    downstream -- the iteration space, the halo scheme, the data space -- is
    derived from plain indexed accesses.
    """
    f = func.function
    grid = f.grid

    terms = []
    for ds, d, fac in zip(grid.dimensions, grid.parent.dimensions,
                          grid.rational_factor, strict=True):
        if d not in target.dimensions:
            continue

        w, offsets, nphases = _phase_table(f, fac)

        # `IntDiv` and `Mod` are non-linear, so they must be evaluated on the
        # *global* index; under MPI the local one differs by this rank's
        # offset. `soff` brings the resulting SubGrid index back into local
        # terms. Without domain decomposition both are literal zero
        poff, soff = grid.ioffsets[d]
        oglb = d + poff

        # `nphases == 1` means every parent point lands exactly on a SubGrid
        # point: a single tap of unit weight, no interpolation
        ph = sympy.Mod(oglb, nphases) if nphases > 1 else None
        terms.append((ds, oglb, soff, ph, w, offsets, fac))

    if not terms:
        return func

    retval = []
    ranges = [range(len(t[5])) for t in terms]
    for taps in product(*ranges):
        weight = sympy.S.One
        mapper = {}
        for (ds, oglb, soff, ph, w, offsets, frac), i in zip(terms, taps,
                                                             strict=True):
            if ph is None:
                if offsets[i] != 0:
                    break
            else:
                weight = weight*w[len(offsets)*ph + i]
            mapper[ds] = _base(oglb, frac) - soff + offsets[i]
        else:
            retval.append(Mul(weight, func.subs(mapper)))

    return Add(*retval)


def _base(d, frac):
    """
    The SubGrid base index for parent index `d`: `(d*q)//p` for a
    `factor = p/q`, in integer arithmetic.
    """
    return IntDiv(frac.denominator*d, frac.numerator)


def _phase_table(f, factor):
    """
    The per-phase interpolation weights for reading `f` at its parent's points
    along one Dimension, as a compile-time table.

    Each phase is a constant fractional shift, so its weights are those of a
    zeroth-order derivative at that `x0` -- the same relocation staggering
    uses, with `f.interp_order` setting the width -- unless `f`'s SubGrid asks
    for windowed sinc, which the FD weights do not express.

    Returns `(weights, offsets, nphases)`: the `Array` holding
    `nphases*len(offsets)` values, the tap offsets from the base index, and the
    number of phases.
    """
    p, q = factor.numerator, factor.denominator

    # Reading the SubGrid at the parent's points inverts the factor, so the
    # phase period is `p`
    nphases = p
    fracs = [sympy.Rational((ph*q) % p, p) for ph in range(nphases)]

    io = f.interp_order
    offsets = list(range(-io//2 + 1, io//2 + 1))

    if f.grid.interpolation == 'sinc':
        b = kaiser_b[io]
        rows = [sinc_weights([o - frac for o in offsets], io//2, b,
                             normalize=True)
                for frac in fracs]
    else:
        rows = [numeric_weights(f, 0, offsets, frac) for frac in fracs]

    values = [sympy.Float(v) for row in rows for v in row]

    # Not a `Weights`: that models a single stencil over a `StencilDimension`,
    # whereas this concatenates one row per phase and is indexed by the phase
    # at run time.
    #
    # Several tables may coexist in one Operator -- a SubGrid's axes may carry
    # different factors, and several SubGrids may be read in one expression --
    # and the emitted C declarations are keyed by name, so the name has to
    # distinguish them by *content*
    digest = sha1(str((values, f.dtype)).encode()).hexdigest()[:8]
    name = f'w{digest}'
    d = CustomDimension(name=f'{name}d', symbolic_min=0,
                        symbolic_max=len(values) - 1,
                        symbolic_size=len(values))
    weights = Array(name=name, dimensions=(d,), shape=(len(values),),
                    dtype=f.dtype, scope='stack', initvalue=values)

    return weights, offsets, nphases

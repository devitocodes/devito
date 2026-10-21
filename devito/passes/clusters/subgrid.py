from fractions import Fraction

import sympy

from devito.finite_differences.interpolation import SubGridInterp
from devito.ir.clusters import Queue
from devito.symbolics import search, uxreplace
from devito.tools import timed_pass
from devito.types import Array, CustomDimension

__all__ = ['subgrid_interp']


@timed_pass()
def subgrid_interp(clusters, sregistry=None, **kwargs):
    """
    Lower a Cluster reading an interpolated SubGrid Function into a plain
    stencil.

    A SubGrid point `i` sits at parent coordinate `i*factor`, so reading it at
    the parent point `x` means evaluating at `x/factor`. With `factor = p/q`
    that coordinate is `x*q/p`, hence

        base = (x*q) // p        frac = ((x*q) % p) / p

    Both are plain integer arithmetic, and `frac` depends on `x` only through
    `x % p` -- so there are `p` phases, and within a phase the tap weights are
    *constants*. Only the weights vary between phases, so they go in a small
    compile-time table indexed by `Mod(x, p)`.

    No phase loop is therefore needed, and nothing is added to the iteration
    space: the SubGrid Dimensions are simply dropped and the inner loops stay
    unit-stride and branch-free.
    """
    return SubGridInterpolation(sregistry).process(clusters)


class SubGridInterpolation(Queue):

    def __init__(self, sregistry=None, **kwargs):
        super().__init__(**kwargs)
        self.sregistry = sregistry

    def callback(self, clusters, prefix):
        if prefix:
            return clusters

        processed = []
        for c in clusters:
            processed.append(_lower(c, self.sregistry))

        return processed


def _lower(cluster, sregistry):
    """Expand the SubGridInterp markers in one Cluster, or return it as is."""
    markers = set()
    for e in cluster.exprs:
        markers.update(search(e, SubGridInterp))

    if not markers:
        return cluster

    ispace = cluster.ispace
    weights = {}       # (factor, dtype) -> (Array, ntaps, nphases)
    subs = {}

    # Blocking may already have run, in which case a parent Dimension appears
    # in the IterationSpace only through its block Dimensions (`y -> y0_blk0`).
    # Promote those back so the membership test below sees the roots
    itdims = set(ispace.promote(lambda d: d.is_Block).itdims)

    for m in markers:
        grid = m.subgrid
        func = m.func_.function

        terms = []
        for ds, d, fac in zip(grid.dimensions, grid.parent.dimensions,
                              grid.factor, strict=True):
            if d not in itdims:
                continue

            frac = Fraction(float(fac)).limit_denominator(10**6)
            key = (float(fac), func.dtype)
            if key not in weights:
                name = sregistry.make_name(prefix='w') if sregistry else f'w{d.name}'
                weights[key] = _phase_weights(fac, func.dtype, name)
            w, ntaps, nphases = weights[key]

            # `nphases == 1` means every parent point lands exactly on a
            # SubGrid point: a single tap of unit weight, no interpolation
            ph = sympy.Mod(d, nphases) if nphases > 1 else None
            terms.append((ds, d, ph, w, ntaps, frac))

        if not terms:
            continue

        # The relocated read: a weighted sum over the taps of every relocated
        # Dimension, with the weights taken from each Dimension's phase row
        subs[m] = _stencil(func, terms)

    exprs = [uxreplace(e, subs) for e in cluster.exprs]

    # The SubGrid Dimensions are no longer iterated: the relocated reads are
    # indexed by the parent's Dimensions alone. An axis of factor 1 is the
    # exception -- there the SubGrid shares the parent's own Dimension, which
    # stays in the IterationSpace
    gone = {ds for m in markers
            for ds, d in zip(m.subgrid.dimensions, m.subgrid.parent.dimensions,
                             strict=True)
            if ds is not d}
    ispace = ispace.project(lambda d: d not in gone)

    return cluster.rebuild(exprs=exprs, ispace=ispace)


def _stencil(func, terms):
    """
    `func` read as a weighted sum over the taps of every relocated Dimension,
    with the weights taken from each Dimension's phase row.
    """
    from itertools import product

    retval = []
    ranges = [range(t[4]) if t[2] is not None else range(1) for t in terms]
    for offsets in product(*ranges):
        weight = sympy.S.One
        indices = {}
        for (ds, d, ph, w, ntaps, frac), o in zip(terms, offsets,
                                                  strict=True):
            if ph is not None:
                weight = weight*w[ntaps*ph + o]
            indices[ds] = _base(d, frac) + o

        # `func` is indexed directly here, which bypasses the halo shift that
        # index lowering applies to ordinary reads -- so add it explicitly
        idxs = [indices.get(d, d) + h for d, h in
                zip(func.dimensions, func._size_nodomain.left, strict=True)]
        retval.append(weight*func.indexed[tuple(idxs)])

    return sympy.Add(*retval)


def _base(d, frac):
    """
    The SubGrid base index for parent index `d`: `(d*q)//p` for a
    `factor = p/q`, in integer arithmetic.

    `d` is the raw iteration Dimension: index lowering adds each Function's
    halo afterwards, so neither halo takes part in the division.
    """
    from devito.symbolics import IntDiv
    return IntDiv(frac.denominator*d, frac.numerator)


def _phase_weights(factor, dtype, name):
    """
    The per-phase tap weights for *reading* a SubGrid of the given `factor` at
    its parent's points, as a compile-time table.

    A SubGrid point sits at parent coordinate `i*factor`, so conversely the
    parent point `x` sits at SubGrid coordinate `x/factor`. With
    `factor = p/q` that is `x*q/p`, hence

        base = (x*q)//p        frac = ((x*q) % p)/p

    `base` is the integer part, and `frac` depends on `x` only through
    `x % p`. So there are `p` phases, and within a phase the two linear
    weights `(1 - frac, frac)` are constants -- an ordinary constant-offset
    stencil, read at `base` and `base + 1`.

    Returns `(weights, ntaps, nphases)`: the `Array` holding `nphases*ntaps`
    values, the taps per phase, and the number of phases.
    """
    frac = Fraction(float(factor)).limit_denominator(10**6)
    p, q = frac.numerator, frac.denominator

    # Reading the SubGrid at the parent's points inverts the factor, so the
    # phase period is `p`
    nphases, ntaps = p, 2

    values = []
    for ph in range(nphases):
        f = float(Fraction((ph*q) % p, p))
        values.extend(sympy.Float(v) for v in (1. - f, f))

    d = CustomDimension(name=f'{name}d', symbolic_min=0,
                        symbolic_max=nphases*ntaps - 1,
                        symbolic_size=nphases*ntaps)
    weights = Array(name=name, dimensions=(d,), shape=(nphases*ntaps,),
                    dtype=dtype, scope='stack', initvalue=values)

    return weights, ntaps, nphases

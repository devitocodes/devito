from devito.tools import Tag


class Property(Tag):

    _KNOWN = []

    def __init__(self, name, val=None):
        super(Property, self).__init__(name, val)
        Property._KNOWN.append(self)


SEQUENTIAL = Property('sequential')
"""A sequential Dimension."""

PARALLEL = Property('parallel')
"""A fully parallel Dimension."""

PARALLEL_INDEP = Property('parallel=')
"""
A fully parallel Dimension, where all dependences have dependence distance
equals to 0 (i.e., the distance vector is '='). This is stronger than PARALLEL.
"""

PARALLEL_IF_ATOMIC = Property('parallel_if_atomic')
"""A parallel Dimension with local reductions. This is weaker than PARALLEL."""

COLLAPSED = lambda i: Property('collapsed', i)
"""Collapsed Dimensions."""

VECTORIZED = Property('vector-dim')
"""A SIMD-vectorized Dimension."""

TILABLE = Property('tilable')
"""A fully parallel Dimension that would benefit from tiling (or "blocking")."""

WRAPPABLE = Property('wrappable')
"""
A modulo-N Dimension (i.e., cycling over i, i+1, i+2, ..., i+N-1) that could
safely be turned into a modulo-K Dimension, with K < N. For example:
u[t+1, ...] = f(u[t, ...]) + u[t-1, ...] --> u[t+1, ...] = f(u[t, ...]) + u[t+1, ...].
"""

ROUNDABLE = Property('roundable')
"""
A Dimension whose upper limit may be rounded up to a multiple of the SIMD
vector length thanks to the presence of enough padding.
"""

AFFINE = Property('affine')
"""
A Dimension used to index into tensor objects only through affine and regular
accesses functions. See :mod:`basic.py` for rigorous definitions of "affine"
and "regular".
"""


def normalize_properties(*args):
    if not args:
        return
    elif len(args) == 1:
        return args[0]

    if any(SEQUENTIAL in p for p in args):
        drop = {PARALLEL, PARALLEL_INDEP, PARALLEL_IF_ATOMIC}
    elif any(PARALLEL_IF_ATOMIC in p for p in args):
        drop = {PARALLEL, PARALLEL_INDEP}
    elif any(PARALLEL_INDEP not in p for p in args):
        drop = {PARALLEL_INDEP}
    else:
        drop = set()

    properties = set()
    for p in args:
        properties.update(p - drop)

    return properties

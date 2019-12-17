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

PARALLEL_IF_ATOMIC = Property('parallel_if_atomic')
"""A parallel Dimension with local reductions."""

COLLAPSED = lambda i: Property('collapsed', i)
"""Collapsing Dimensions."""

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


def normalize_properties(properties):
    properties = set(properties)

    if SEQUENTIAL in properties:
        properties -= {PARALLEL, PARALLEL_IF_ATOMIC}
    elif PARALLEL_IF_ATOMIC in properties:
        properties -= {PARALLEL}

    return properties


class HaloSpotProperty(Tag):
    pass


OVERLAPPABLE = HaloSpotProperty('overlappable')
"""The HaloSpot supports computation-communication overlap."""


def hoistable(i):
    """
    The HaloSpot can be squashed with a previous HaloSpot as all data dependences
    would still be honored.
    """
    return HaloSpotProperty('hoistable', i)


def useless(i):
    """
    The HaloSpot can be ignored as a halo update at this point would be completely
    useless.
    """
    return HaloSpotProperty('useless', i)

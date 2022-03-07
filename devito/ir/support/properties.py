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
"""
A parallel Dimension that requires atomic reductions. This is weaker than PARALLEL.
"""

PARALLEL_IF_PVT = Property('parallel_if_private')
"""
A parallel Dimension that requires all compiler-generated AbstractFunctions be
privatized at the thread level. This is weaker than PARALLEL.
"""

PARALLEL_INDEP = Property('parallel=')
"""
A fully parallel Dimension, where all dependences have dependence distance
equals to 0 (i.e., the distance vector is '=').
"""

COLLAPSED = lambda i: Property('collapsed', i)
"""Collapsed Dimensions."""

VECTORIZED = Property('vector-dim')
"""A SIMD-vectorized Dimension."""

TILABLE = Property('tilable')
"""A fully parallel Dimension that would benefit from tiling (or "blocking")."""

SKEWABLE = Property('skewable')
"""A fully parallel Dimension that would benefit from wavefront/skewed tiling."""

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

SEPARABLE = Property('separable')
"""
This property is thought for Hyperplanes, that is collections of Dimensions,
rather than individual Dimensions. A SEPARABLE Hyperplane defines an iteration
space that could be separated into multiple smaller hyperplanes to avoid
iterating over the unnecessary hypercorners. For example, the following
SEPARABLE 4x4 plane, that as such needs no iteration over the corners `*`,

    * a a *
    b b b b
    b b b b
    * c c *

could be separated into three one-dimensional iteration spaces

      a a

      b b b b
      b b b b

      c c
"""


def normalize_properties(*args):
    if not args:
        return
    elif len(args) == 1:
        return args[0]

    # Some properties are incompatible, such as SEQUENTIAL and PARALLEL where
    # SEQUENTIAL clearly takes precedence. The precedence chain, from the least
    # to the most restrictive property, is:
    # SEQUENTIAL > PARALLEL_IF_* > PARALLEL > PARALLEL_INDEP
    if any(SEQUENTIAL in p for p in args):
        drop = {PARALLEL, PARALLEL_INDEP, PARALLEL_IF_ATOMIC, PARALLEL_IF_PVT}
    elif any(PARALLEL_IF_ATOMIC in p for p in args):
        drop = {PARALLEL, PARALLEL_INDEP}
    elif any(PARALLEL_IF_PVT in p for p in args):
        drop = {PARALLEL}
    elif any(PARALLEL_INDEP not in p for p in args):
        drop = {PARALLEL_INDEP}
    else:
        drop = set()

    # SEPARABLE <=> all are SEPARABLE
    if not all(SEPARABLE in p for p in args):
        drop.add(SEPARABLE)

    properties = set()
    for p in args:
        properties.update(p - drop)

    return properties


def relax_properties(properties):
    return frozenset(properties - {PARALLEL_INDEP, ROUNDABLE})

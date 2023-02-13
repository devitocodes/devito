from devito.tools import Tag, as_tuple, frozendict


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

INBOUND = Property('inbound')
"""
A Dimension defining an iteration space that is guaranteed to generate in-bounds
array accesses, typically through the use of custom conditionals in the body. This
is used for iteration spaces that are larger than the data space.
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


class Properties(frozendict):

    """
    A mapper {Dimension -> {properties}}.
    """

    def add(self, dims, properties=None):
        m = dict(self)
        for d in as_tuple(dims):
            m[d] = set(self.get(d, [])) | set(as_tuple(properties))
        return Properties(m)

    def drop(self, dims, properties=None):
        m = dict(self)
        for d in as_tuple(dims):
            if properties is None:
                m.pop(d, None)
            else:
                m[d] = self[d] - set(as_tuple(properties))
        return Properties(m)

    def parallelize(self, dims):
        m = dict(self)
        for d in as_tuple(dims):
            v = set(self.get(d, []))
            v.difference_update({PARALLEL_IF_PVT, PARALLEL_IF_ATOMIC, SEQUENTIAL})
            v.add(PARALLEL)
            m[d] = v
        return Properties(m)

    def affine(self, dims):
        m = dict(self)
        for d in as_tuple(dims):
            m[d] = set(self.get(d, [])) | {AFFINE}
        return Properties(m)

    def sequentialize(self, dims):
        m = dict(self)
        for d in as_tuple(dims):
            m[d] = normalize_properties(set(self.get(d, [])), {SEQUENTIAL})
        return Properties(m)

    def inbound(self, dims):
        m = dict(self)
        for d in as_tuple(dims):
            m[d] = set(m.get(d, [])) | {INBOUND}
        return Properties(m)

    def is_parallel(self, dims):
        return any(len(self[d] & {PARALLEL, PARALLEL_INDEP}) > 0
                   for d in as_tuple(dims))

    def is_parallel_relaxed(self, dims):
        items = {PARALLEL, PARALLEL_INDEP, PARALLEL_IF_ATOMIC, PARALLEL_IF_PVT}
        return any(len(self[d] & items) > 0 for d in as_tuple(dims))

    def is_affine(self, dims):
        return any(AFFINE in self.get(d, ()) for d in as_tuple(dims))

    def is_inbound(self, dims):
        return any(INBOUND in self.get(d, ()) for d in as_tuple(dims))

    def is_sequential(self, dims):
        return any(SEQUENTIAL in self.get(d, ()) for d in as_tuple(dims))

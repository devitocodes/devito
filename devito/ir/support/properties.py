from devito.tools import Tag, as_tuple, frozendict


class Property(Tag):

    _KNOWN = []

    def __init__(self, name, val=None):
        super().__init__(name, val)
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

VECTORIZED = Property('vector-dim')
"""A SIMD-vectorized Dimension."""

TILABLE = Property('tilable')
"""A fully parallel Dimension that would benefit from tiling (or "blocking")."""

TILABLE_SMALL = Property('tilable*')
"""
Like TILABLE, but it would benefit from relatively small block, since the
iteration space is likely to be very small.
"""

SKEWABLE = Property('skewable')
"""A fully parallel Dimension that would benefit from wavefront/skewed tiling."""

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

PREFETCHABLE = Property('prefetchable')
"""
A Dimension along which prefetching is feasible and beneficial.
"""

PREFETCHABLE_SHM = Property('prefetchable-shm')
"""
A Dimension along which shared-memory prefetching is feasible and beneficial.
"""


# Bundles
PARALLELS = {PARALLEL, PARALLEL_INDEP, PARALLEL_IF_ATOMIC, PARALLEL_IF_PVT}
TILABLES = {TILABLE, TILABLE_SMALL}


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
        drop = PARALLELS
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
    return frozenset(properties - {PARALLEL_INDEP})


def tailor_properties(properties, ispace):
    """
    Create a new Properties object off `properties` that retains all and only
    the iteration dimensions in `ispace`.
    """
    for i in properties:
        for d in as_tuple(i):
            if d not in ispace.itdims:
                properties = properties.drop(d)

    for d in ispace.itdims:
        properties = properties.add(d)

    return properties


def update_properties(properties, exprs):
    """
    Create a new Properties object off `properties` augmented with properties
    discovered from `exprs` or with properties removed if they are incompatible
    with `exprs`.
    """
    exprs = as_tuple(exprs)

    if not exprs:
        return properties

    # Auto-detect prefetchable Dimensions
    dims = set()
    flag = False
    for e in as_tuple(exprs):
        w, r = e.args

        # Ensure it's in the form `Indexed = Indexed`
        try:
            wf, rf = w.function, r.function
        except AttributeError:
            break

        if not wf._mem_shared:
            break
        dims.update({d.parent for d in wf.dimensions if d.parent in properties})

        if not rf._mem_heap:
            break
    else:
        flag = True

    if flag:
        properties = properties.prefetchable_shm(dims)
    else:
        properties = properties.drop(properties=PREFETCHABLE_SHM)

    return properties


class Properties(frozendict):

    """
    A mapper {Dimension -> {properties}}.
    """

    @property
    def dimensions(self):
        return tuple(self)

    def add(self, dims, properties=None):
        m = dict(self)
        for d in as_tuple(dims):
            m[d] = set(self.get(d, [])) | set(as_tuple(properties))
        return Properties(m)

    def filter(self, key):
        m = {d: v for d, v in self.items() if key(d)}
        return Properties(m)

    def drop(self, dims=None, properties=None):
        if dims is None:
            dims = list(self)
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

    def sequentialize(self, dims=None):
        if dims is None:
            dims = list(self)
        m = dict(self)
        for d in as_tuple(dims):
            m[d] = normalize_properties(set(self.get(d, [])), {SEQUENTIAL})
        return Properties(m)

    def prefetchable(self, dims, v=PREFETCHABLE):
        m = dict(self)
        for d in as_tuple(dims):
            m[d] = self.get(d, set()) | {v}
        return Properties(m)

    def prefetchable_shm(self, dims):
        return self.prefetchable(dims, PREFETCHABLE_SHM)

    def block(self, dims, kind='default'):
        if kind == 'default':
            p = TILABLE
        elif kind == 'small':
            p = TILABLE_SMALL
        else:
            raise ValueError
        m = dict(self)
        for d in as_tuple(dims):
            m[d] = set(self.get(d, [])) | {p}
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
        return any(len(self[d] & PARALLELS) > 0 for d in as_tuple(dims))

    def is_parallel_atomic(self, dims):
        return any(PARALLEL_IF_ATOMIC in self.get(d, ()) for d in as_tuple(dims))

    def is_affine(self, dims):
        return any(AFFINE in self.get(d, ()) for d in as_tuple(dims))

    def is_inbound(self, dims):
        return any(INBOUND in self.get(d, ()) for d in as_tuple(dims))

    def is_sequential(self, dims):
        return any(SEQUENTIAL in self.get(d, ()) for d in as_tuple(dims))

    def is_blockable(self, d):
        return bool(self.get(d, set()) & {TILABLE, TILABLE_SMALL})

    def is_blockable_small(self, d):
        return TILABLE_SMALL in self.get(d, set())

    def is_prefetchable(self, dims=None, v=PREFETCHABLE):
        if dims is None:
            dims = list(self)
        return any(v in self.get(d, set()) for d in as_tuple(dims))

    def is_prefetchable_shm(self, dims=None):
        return self.is_prefetchable(dims, PREFETCHABLE_SHM)

    @property
    def nblockable(self):
        return sum([self.is_blockable(d) for d in self])

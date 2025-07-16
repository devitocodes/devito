from sympy import sympify

from devito.finite_differences.differentiable import IndexSum
from devito.ir.clusters import Queue
from devito.ir.support import (AFFINE, PARALLEL, PARALLEL_IF_ATOMIC,
                               PARALLEL_IF_PVT, SKEWABLE, TILABLES, Interval,
                               IntervalGroup, IterationSpace, Scope)
from devito.passes import is_on_device
from devito.symbolics import search, uxreplace, xreplace_indices
from devito.tools import (UnboundedMultiTuple, UnboundTuple, as_mapper, as_tuple,
                          filter_ordered, flatten, is_integer)
from devito.types import BlockDimension

__all__ = ['blocking']


def blocking(clusters, sregistry, options):
    """
    Loop blocking to improve data locality.

    Parameters
    ----------
    clusters : tuple of Clusters
        Input Clusters, subject of the optimization pass.
    options : dict
        The optimization options.
        * `blockrelax`: use/ignore heuristics to apply blocking to
          potentially inexpensive loop nests.
        * `blockinner`: enable/disable loop blocking along the
          innermost loop.
        * `blocklevels`: 1 => classic loop blocking; 2 for two-level
          hierarchical blocking.
        * `skewing`: enable/disable loop skewing.

    Examples
    -------
    A typical use case, e.g.

                    Classical   +blockinner  2-level Hierarchical
    for x            for xb        for xb         for xbb
      for y    -->    for yb        for yb         for ybb
        for z          for x         for zb         for xb
                        for y         for x          for yb
                         for z         for y          for x
                                        for z          for y
                                                        for z

    Notes
    ------
    In case of skewing, if 'blockinner' is enabled, the innermost loop is also skewed.
    """
    if options['blockrelax']:
        if options['blockrelax'] == 'device-aware':
            analyzer = AnalyzeDeviceAwareBlocking(options)
        else:
            analyzer = AnalyzeBlocking(options)
    else:
        analyzer = AnalyzeHeuristicBlocking(options)
    clusters = analyzer.process(clusters)

    if options['skewing']:
        clusters = AnalyzeSkewing().process(clusters)

    if options['blocklevels'] > 0:
        clusters = SynthesizeBlocking(sregistry, options).process(clusters)

    if options['skewing']:
        clusters = SynthesizeSkewing(options).process(clusters)

    return clusters


class AnayzeBlockingBase(Queue):

    """
    Encode the TILABLE property.
    """

    def __init__(self, options):
        super().__init__()

        self.skewing = options['skewing']

    def process(self, clusters):
        return self._process_fatd(clusters, 1)

    def _process_fatd(self, clusters, level, prefix=None):
        # Truncate recursion in case of TILABLE, non-perfect sub-nests, as
        # it's an unsupported case
        if prefix:
            d = prefix[-1].dim

            if any(c.properties.is_blockable(d) for c in clusters) and \
               len({c.ispace[:level] for c in clusters}) > 1:
                return clusters

        return super()._process_fatd(clusters, level, prefix)

    def _has_data_reuse(self, cluster):
        # A sufficient condition for the existance of data reuse in `cluster`
        # is that the same Function is accessed twice at the same memory location,
        # which translates into the existance of any Relation accross Indexeds
        if any(r.function.is_AbstractFunction for r in cluster.scope.r_gen()):
            return True
        if search(cluster.exprs, IndexSum):
            return True

        # If it's a reduction operation a la matrix-matrix multiply, two Indexeds
        # might be enough
        if any(PARALLEL_IF_ATOMIC in p for p in cluster.properties.values()):
            return True

        # If we are going to skew, then we might exploit reuse along an
        # otherwise SEQUENTIAL Dimension
        if self.skewing:
            return True

        return False

    def _has_short_trip_count(self, d):
        # Iteration spaces of statically known size are always small, at
        # most a few tens of unit, so they wouldn't benefit from blocking
        return is_integer(d.symbolic_size)


class AnalyzeBlocking(AnayzeBlockingBase):

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim
        if self._has_short_trip_count(d):
            return clusters

        processed = []
        for c in clusters:
            if not {PARALLEL,
                    PARALLEL_IF_ATOMIC,
                    PARALLEL_IF_PVT}.intersection(c.properties[d]):
                return clusters

            # Pointless if there's no data reuse
            if not self._has_data_reuse(c):
                return clusters

            # All good so far, `d` is actually TILABLE
            processed.append(c.rebuild(properties=c.properties.block(d)))

        return processed


class AnalyzeDeviceAwareBlocking(AnalyzeBlocking):

    def __init__(self, options):
        super().__init__(options)

        self.gpu_fit = options.get('gpu-fit', ())

    def _make_key_hook(self, cluster, level):
        return (is_on_device(cluster.functions, self.gpu_fit),)

    def _has_atomic_blockable_dim(self, cluster, d):
        return any(cluster.properties.is_parallel_atomic(i)
                   for i in set(cluster.ispace.itdims) - {d})

    def _has_enough_large_blockable_dims(self, cluster, d, nested=False):
        if nested:
            _, ispace = cluster.ispace.split(d)
            dims = set(ispace.itdims)
        else:
            ispace = cluster.ispace
            dims = set(cluster.ispace.itdims) - {d}
        return len([i for i in dims
                    if (cluster.properties.is_parallel_relaxed(i) and
                        not self._has_short_trip_count(i))]) >= 3

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        processed = []
        for c in clusters:
            if not c.properties.is_parallel_relaxed(d):
                return clusters

            if is_on_device(c.functions, self.gpu_fit):
                if self._has_short_trip_count(d):
                    if self._has_atomic_blockable_dim(c, d):
                        # Optimization: minimize number of parallel reductions
                        # if we think there's already enough parallelism around
                        return clusters
                    elif self._has_enough_large_blockable_dims(c, d):
                        # Optimization: pointless, from a performance standpoint,
                        # to have more than three large blockable Dimensions
                        return clusters

                if self._has_enough_large_blockable_dims(c, d, nested=True):
                    # Virtually all device programming models forbid parallelism
                    # along more than three dimensions
                    return clusters

                if any(self._has_short_trip_count(i) for i in c.ispace.itdims):
                    properties = c.properties.block(d, 'small')
                elif self._has_data_reuse(c):
                    properties = c.properties.block(d)
                else:
                    properties = c.properties.block(d, 'small')

            elif self._has_data_reuse(c):
                properties = c.properties.block(d)

            else:
                return clusters

            processed.append(c.rebuild(properties=properties))

        return processed


class AnalyzeHeuristicBlocking(AnayzeBlockingBase):

    def __init__(self, options):
        super().__init__(options)

        self.inner = options['blockinner']

    def process(self, clusters):
        clusters = super().process(clusters)

        # Heuristic: if there aren't at least two TILABLE Dimensions, drop it
        processed = []
        for c in clusters:
            if c.properties.nblockable > 1:
                processed.append(c)
            else:
                properties = c.properties.drop(properties=TILABLES)
                processed.append(c.rebuild(properties=properties))

        return processed

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim
        if self._has_short_trip_count(d):
            return clusters

        # Pointless if there's no data reuse
        if all(not self._has_data_reuse(c) for c in clusters):
            return clusters

        # Heuristic: if all Clusters operate on local SubDimensions, then it means
        # that all IterationSpaces are tiny, hence we can skip
        if all(any(i.dim.is_Sub and i.dim.local for i in c.ispace) for c in clusters):
            return clusters

        processed = []
        for c in clusters:
            # PARALLEL* and AFFINE are necessary conditions
            if AFFINE not in c.properties[d] or \
               not ({PARALLEL, PARALLEL_IF_PVT} & c.properties[d]):
                return clusters

            # Heuristic: innermost Dimensions may be ruled out a-priori
            is_inner = d is c.ispace[-1].dim
            if is_inner and not self.inner:
                return clusters

            # Heuristic: TILABLE not worth it if not within a SEQUENTIAL Dimension
            if not any(c.properties.is_sequential(i.dim) for i in prefix[:-1]):
                return clusters

            processed.append(c.rebuild(properties=c.properties.block(d)))

        if len(clusters) > 1:
            # Heuristic: same as above if it induces dynamic bounds
            exprs = flatten(c.exprs for c in as_tuple(clusters))
            scope = Scope(exprs)
            if any(i.is_lex_non_stmt for i in scope.d_all_gen()):
                return clusters
        else:
            # Just avoiding potentially expensive checks
            pass

        return processed


class AnalyzeSkewing(Queue):

    """
    Encode the SKEWABLE Dimensions.
    """

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        processed = []
        for c in clusters:
            if not c.properties.is_blockable(d):
                return clusters

            processed.append(c.rebuild(properties=c.properties.add(d, SKEWABLE)))

        return processed


class SynthesizeBlocking(Queue):

    _q_guards_in_key = True

    template = "%s%d_blk%s"

    def __init__(self, sregistry, options):
        self.sregistry = sregistry

        self.levels = options['blocklevels']
        self.par_tile = options['par-tile']

        # Track the BlockDimensions created so far so that we can reuse them
        # in case of Clusters that are different but share the same number of
        # stencil points
        self.mapper = {}

        super().__init__()

    def process(self, clusters):
        # A tool to unroll the explicit integer block shapes, should there be any
        if self.par_tile:
            blk_size_gen = BlockSizeGenerator(self.par_tile)
        else:
            blk_size_gen = None

        return self._process_fdta(clusters, 1, blk_size_gen=blk_size_gen)

    def _derive_block_dims(self, clusters, prefix, d, blk_size_gen):
        if blk_size_gen is not None:
            step = sympify(blk_size_gen.next(prefix, d, clusters))
        else:
            # This will result in a parametric step, e.g. `x0_blk0_size`
            step = None

        # Can I reuse existing BlockDimensions to avoid a proliferation of steps?
        k = stencil_footprint(clusters, d)
        if step is None:
            try:
                return self.mapper[k]
            except KeyError:
                pass

        base = self.sregistry.make_name(prefix=d.root.name)

        name = self.sregistry.make_name(prefix=f"{base}_blk")
        bd = BlockDimension(name, d, d.symbolic_min, d.symbolic_max, step)
        step = bd.step
        block_dims = [bd]

        for _ in range(1, self.levels):
            name = self.sregistry.make_name(prefix=f"{base}_blk")
            bd = BlockDimension(name, bd, bd, bd + bd.step - 1, size=step)
            block_dims.append(bd)

        bd = BlockDimension(d.name, bd, bd, bd + bd.step - 1, 1, size=step)
        block_dims.append(bd)

        retval = self.mapper[k] = tuple(block_dims), bd

        return retval

    def callback(self, clusters, prefix, blk_size_gen=None):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        if not any(c.properties.is_blockable(d) for c in clusters):
            return clusters

        try:
            block_dims, bd = self._derive_block_dims(
                clusters, prefix, d, blk_size_gen
            )
        except StopIteration:
            return clusters

        processed = []
        for c in clusters:
            if c.properties.is_blockable(d):
                ispace = decompose(c.ispace, d, block_dims)

                # Use the innermost BlockDimension in place of `d`
                subs = {d: bd}
                exprs = [uxreplace(e, subs) for e in c.exprs]
                guards = {subs.get(i, i): v for i, v in c.guards.items()}

                # The new Cluster properties -- TILABLE is dropped after blocking
                properties = c.properties.drop(d)
                properties = properties.add(block_dims, c.properties[d] - TILABLES)

                processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                           guards=guards, properties=properties))
            else:
                processed.append(c)

        return processed


def stencil_footprint(clusters, d):
    """
    Compute the number of stencil points in the given Dimension `d` across the
    provided Clusters.
    """
    # The following would be an approximation in the case of irregular Clusters,
    # but if we're here it means the Clusters are likely regular, so it's fine
    indexeds = set().union(*[c.scope.indexeds for c in clusters])
    indexeds = [i for i in indexeds if d._defines & set(i.dimensions)]

    # Distinguish between footprints pertaining to different Functions
    mapper = as_mapper(indexeds, lambda i: i.function)
    n = tuple(sorted(len(v) for v in mapper.values()))

    return (d, n)


def decompose(ispace, d, block_dims):
    """
    Create a new IterationSpace in which the `d` Interval is decomposed
    into a hierarchy of Intervals over ``block_dims``.
    """
    # Create the new Intervals
    intervals = []
    for i in ispace:
        if i.dim is d:
            intervals.append(i.switch(block_dims[0]))
            intervals.extend([i.switch(bd).zero() for bd in block_dims[1:]])
        else:
            intervals.append(i)

    # Create the intervals relations
    # 1: `bbd > bd`
    relations = [block_dims]

    # 2: Suitably replace `d` with all `bd`'s
    for r in ispace.relations:
        if not d._defines.intersection(r):
            relations.append(r)
            continue

        for bd in block_dims:
            # Avoid e.g. `x > yb`
            if any(i._depth < bd._depth for i in r if i.is_Block):
                continue

            # E.g., `r=(z, i0z)` -> `[i0z0_blk0, i0z0_blk0]`
            v = [bd if i in d._defines else i for i in r]

            # E.g., `v=[i0z0_blk0, i0z0_blk0]` -> `v=(i0z0_blk0,)`
            v = tuple(filter_ordered(v))

            relations.append(v)

    # 3: Make sure BlockDimensions at same depth stick next to each other
    # E.g., `(t, xbb, ybb, xb, yb, x, y)`, and NOT e.g. `(t, xbb, xb, x, ybb, ...)`
    # NOTE: this is perfectly legal since:
    # TILABLE => (perfect nest & PARALLEL) => interchangeable
    for i in ispace.itdims:
        if not i.is_Block:
            continue
        for bd in block_dims:
            if i._depth < bd._depth:
                # E.g. `(zb, y)`
                relations.append((i, bd))
            elif i._depth == bd._depth:
                # E.g. `(y, z)` (i.e., honour input ordering)
                relations.append((bd, i))

    intervals = IntervalGroup(intervals, relations=relations)

    sub_iterators = dict(ispace.sub_iterators)
    sub_iterators.pop(d, None)
    sub_iterators.update({bd: () for bd in block_dims[:-1]})
    sub_iterators.update({block_dims[-1]: ispace.sub_iterators[d]})

    directions = dict(ispace.directions)
    directions.pop(d)
    directions.update({bd: ispace.directions[d] for bd in block_dims})

    return IterationSpace(intervals, sub_iterators, directions)


class BlockSizeGenerator:

    """
    A wrapper for several UnboundedMultiTuples.
    """

    def __init__(self, par_tile):
        self.tip = -1
        self.umt = par_tile

        if par_tile.is_multi:
            # The user has supplied one specific par-tile per blocked nest
            self.umt_small = par_tile
            self.umt_sparse = par_tile
            self.umt_reduce = par_tile
        else:
            # Special case 1: a smaller par-tile to avoid under-utilizing
            # computational resources when the iteration spaces are too small
            self.umt_small = UnboundedMultiTuple(par_tile.default)

            # Special case 2: par-tiles for iteration spaces that must be fully
            # blocked for correctness
            if par_tile.sparse:
                self.umt_sparse = UnboundTuple(*par_tile.sparse, 1)
            elif len(par_tile) == 1:
                self.umt_sparse = UnboundTuple(*par_tile[0], 1)
            else:
                self.umt_sparse = UnboundTuple(*par_tile.default, 1)

            if par_tile.reduce:
                self.umt_reduce = UnboundTuple(*par_tile.reduce, 1)
            elif len(par_tile) == 1:
                self.umt_reduce = UnboundTuple(*par_tile[0], 1)
            else:
                self.umt_reduce = UnboundTuple(*par_tile.default, 1)

    def next(self, prefix, d, clusters):
        # If a whole new set of Dimensions, move the tip -- this means `clusters`
        # at `d` represents a new loop nest or kernel
        x = any(i.dim.is_Block for i in flatten(c.ispace for c in clusters))
        if not x:
            self.tip += 1

        # Correctness -- enforce blocking where necessary.
        # See also issue #276:PRO
        if any(c.properties.is_parallel_atomic(d) for c in clusters):
            if any(c.is_sparse for c in clusters):
                if not x:
                    self.umt_sparse.iter()
                return self.umt_sparse.next()
            else:
                if not x:
                    self.umt_reduce.iter()
                return self.umt_reduce.next()

        # Performance heuristics -- use a smaller par-tile
        if all(c.properties.is_blockable_small(d) for c in clusters):
            if not x:
                self.umt_small.iter()
            return self.umt_small.next()

        if x:
            item = self.umt.curitem()
        else:
            # We can't `self.umt.iter()` because we might still want to
            # fallback to `self.umt_small`
            item = self.umt.nextitem()

        # Handle user-provided rules
        # TODO: This is also rudimentary
        if item.rule is None:
            umt = self.umt
        elif is_integer(item.rule):
            if item.rule == self.tip:
                umt = self.umt
            else:
                umt = self.umt_small
                if not x:
                    umt.iter()
        else:
            if item.rule in {d.name for d in prefix.itdims}:
                umt = self.umt
            else:
                # This is like "pattern unmatched" -- fallback to `umt_small`
                umt = self.umt_small
                if not x:
                    umt.iter()

        return umt.next()


class SynthesizeSkewing(Queue):

    """
    Construct a new sequence of clusters with skewed expressions and
    iteration spaces.

    Notes
    -----
    This transformation is applying loop skewing to derive the
    wavefront method of execution of nested loops. Loop skewing is
    a simple transformation of loop bounds and is combined with loop
    interchanging to generate the wavefront [1]_.

    .. [1] Wolfe, Michael. "Loops skewing: The wavefront method revisited."
    International Journal of Parallel Programming 15.4 (1986): 279-293.

    Examples:

    .. code-block:: python

        for i = 2, n-1
            for j = 2, m-1
                a[i,j] = (a[a-1,j] + a[i,j-1] + a[i+1,j] + a[i,j+1]) / 4

    to

    .. code-block:: python

        for i = 2, n-1
            for j = 2+i, m-1+i
                a[i,j-i] = (a[a-1,j-i] + a[i,j-1-i] + a[i+1,j-i] + a[i,j+1-i]) / 4

    """

    def __init__(self, options):
        self.skewinner = bool(options['blockinner'])

        super().__init__()

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        processed = []
        for c in clusters:
            if SKEWABLE not in c.properties[d]:
                return clusters

            skew_dims = {i.dim for i in c.ispace
                         if c.properties.is_sequential(i.dim)}
            if len(skew_dims) > 1:
                return clusters
            skew_dim = skew_dims.pop()

            # Since we are here, prefix is skewable and nested under a
            # SEQUENTIAL loop
            intervals = []
            for i in c.ispace:
                if i.dim is d and (not d.is_Block or d._depth == 1):
                    intervals.append(Interval(d, skew_dim, skew_dim))
                else:
                    intervals.append(i)
            intervals = IntervalGroup(intervals, relations=c.ispace.relations)
            ispace = IterationSpace(intervals, c.ispace.sub_iterators,
                                    c.ispace.directions)

            exprs = xreplace_indices(c.exprs, {d: d - skew_dim})
            processed.append(c.rebuild(exprs=exprs, ispace=ispace))

        return processed

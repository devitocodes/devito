from sympy import sympify

from devito.ir.clusters import Queue
from devito.ir.support import (AFFINE, PARALLEL, PARALLEL_IF_ATOMIC, PARALLEL_IF_PVT,
                               SEQUENTIAL, SKEWABLE, TILABLE, Interval, IntervalGroup,
                               IterationSpace, Scope)
from devito.passes import is_on_device
from devito.symbolics import uxreplace, xreplace_indices
from devito.tools import UnboundedMultiTuple, as_tuple, flatten, is_integer
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

            if any(TILABLE in c.properties[d] for c in clusters) and \
               len({c.ispace[:level] for c in clusters}) > 1:
                return clusters

        return super()._process_fatd(clusters, level, prefix)

    def _has_data_reuse(self, cluster):
        # A sufficient condition for the existance of data reuse in `cluster`
        # is that the same Function is accessed twice via two different Indexeds
        seen = set()
        for i in cluster.scope.indexeds:
            if i.function in seen:
                return True
            else:
                seen.add(i.function)

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
            processed.append(c.rebuild(properties=c.properties.add(d, TILABLE)))

        return processed


class AnalyzeDeviceAwareBlocking(AnalyzeBlocking):

    def __init__(self, options):
        super().__init__(options)

        self.gpu_fit = options.get('gpu-fit', ())

    def _make_key_hook(self, cluster, level):
        return (is_on_device(cluster.functions, self.gpu_fit),)

    def _has_data_reuse(self, cluster):
        if is_on_device(cluster.functions, self.gpu_fit):
            return True
        else:
            return super()._has_data_reuse(cluster)


class AnalyzeHeuristicBlocking(AnayzeBlockingBase):

    def __init__(self, options):
        super().__init__(options)

        self.inner = options['blockinner']

    def process(self, clusters):
        clusters = super().process(clusters)

        # Heuristic: if there aren't at least two TILABLE Dimensions, drop it
        processed = []
        for c in clusters:
            ntilable = len([TILABLE for v in c.properties.values() if TILABLE in v])
            if ntilable > 1:
                processed.append(c)
            else:
                properties = {d: v - {TILABLE} for d, v in c.properties.items()}
                processed.append(c.rebuild(properties=properties))

        return processed

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim
        if self._has_short_trip_count(d):
            return clusters

        processed = []
        for c in clusters:
            # PARALLEL* and AFFINE are necessary conditions
            if AFFINE not in c.properties[d] or \
               not ({PARALLEL, PARALLEL_IF_PVT} & c.properties[d]):
                return clusters

            # Pointless if there's no data reuse
            if not self._has_data_reuse(c):
                return clusters

            # Heuristic: innermost Dimensions may be ruled out a-priori
            is_inner = d is c.ispace[-1].dim
            if is_inner and not self.inner:
                return clusters

            # Heuristic: TILABLE not worth it if not within a SEQUENTIAL Dimension
            if not any(SEQUENTIAL in c.properties[i.dim] for i in prefix[:-1]):
                return clusters

            # Heuristic: same as above if there's a local SubDimension
            if any(i.dim.is_Sub and i.dim.local for i in c.ispace):
                return clusters

            processed.append(c.rebuild(properties=c.properties.add(d, TILABLE)))

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
            if TILABLE not in c.properties[d]:
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

        super().__init__()

    def process(self, clusters):
        # A tool to unroll the explicit integer block shapes, should there be any
        if self.par_tile:
            blk_size_gen = BlockSizeGenerator(*self.par_tile)
        else:
            blk_size_gen = None

        return self._process_fdta(clusters, 1, blk_size_gen=blk_size_gen)

    def callback(self, clusters, prefix, blk_size_gen=None):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        if not any(TILABLE in c.properties[d] for c in clusters):
            return clusters

        # Create the block Dimensions (in total `self.levels` Dimensions)
        base = self.sregistry.make_name(prefix=d.name)

        if blk_size_gen is not None:
            # By passing a suitable key to `next` we ensure that we pull the
            # next par-tile entry iff we're now blocking an unseen TILABLE nest
            try:
                step = sympify(blk_size_gen.next(clusters))
            except StopIteration:
                return clusters
        else:
            # This will result in a parametric step, e.g. `x0_blk0_size`
            step = None

        name = self.sregistry.make_name(prefix="%s_blk" % base)
        bd = BlockDimension(name, d, d.symbolic_min, d.symbolic_max, step)
        step = bd.step
        block_dims = [bd]

        for _ in range(1, self.levels):
            name = self.sregistry.make_name(prefix="%s_blk" % base)
            bd = BlockDimension(name, bd, bd, bd + bd.step - 1, size=step)
            block_dims.append(bd)

        bd = BlockDimension(d.name, bd, bd, bd + bd.step - 1, 1, size=step)
        block_dims.append(bd)

        processed = []
        for c in clusters:
            if TILABLE in c.properties[d]:
                ispace = decompose(c.ispace, d, block_dims)

                # Use the innermost BlockDimension in place of `d`
                exprs = [uxreplace(e, {d: bd}) for e in c.exprs]

                # The new Cluster properties -- TILABLE is dropped after blocking
                properties = dict(c.properties)
                properties.pop(d)
                properties.update({bd: c.properties[d] - {TILABLE} for bd in block_dims})

                processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                           properties=properties))
            else:
                processed.append(c)

        return processed


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
    relations = [tuple(block_dims)]

    # 2: Suitably replace `d` with all `bd`'s
    for r in ispace.relations:
        if not d._defines.intersection(r):
            relations.append(r)
            continue

        for bd in block_dims:
            # Avoid e.g. `x > yb`
            if any(i._depth < bd._depth for i in r if i.is_Block):
                continue

            relations.append(tuple(bd if i in d._defines else i for i in r))

    # 3: Make sure BlockDimensions at same depth stick next to each other
    # E.g., `(t, xbb, ybb, xb, yb, x, y)`, and NOT e.g. `(t, xbb, xb, x, ybb, ...)`
    # NOTE: this is perfectly legal since:
    # TILABLE => (perfect nest & PARALLEL) => interchangeable
    for i in ispace.itdimensions:
        if not i.is_Block:
            continue
        for bd in block_dims:
            if i._depth < bd._depth:
                relations.append((i, bd))

    intervals = IntervalGroup(intervals, relations=relations)

    sub_iterators = dict(ispace.sub_iterators)
    sub_iterators.pop(d, None)
    sub_iterators.update({bd: () for bd in block_dims[:-1]})
    sub_iterators.update({block_dims[-1]: ispace.sub_iterators[d]})

    directions = dict(ispace.directions)
    directions.pop(d)
    directions.update({bd: ispace.directions[d] for bd in block_dims})

    return IterationSpace(intervals, sub_iterators, directions)


class BlockSizeGenerator(UnboundedMultiTuple):

    def next(self, clusters):
        if not any(i.dim.is_Block for i in flatten(c.ispace for c in clusters)):
            self.iter()
        return super().next()


class SynthesizeSkewing(Queue):

    """
    Construct a new sequence of clusters with skewed expressions and iteration spaces.

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

            skew_dims = {i.dim for i in c.ispace if SEQUENTIAL in c.properties[i.dim]}
            if len(skew_dims) > 1:
                return clusters
            skew_dim = skew_dims.pop()

            # Since we are here, prefix is skewable and nested under a SEQUENTIAL loop
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

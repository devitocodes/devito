from collections import Counter

from devito.ir.clusters import Queue
from devito.ir.support import (AFFINE, PARALLEL, PARALLEL_IF_ATOMIC, PARALLEL_IF_PVT,
                               SEQUENTIAL, SKEWABLE, TILABLE, Interval, IntervalGroup,
                               IterationSpace, Scope)
from devito.symbolics import uxreplace
from devito.tools import as_tuple, flatten
from devito.types import BlockDimension

from devito.symbolics import xreplace_indices

__all__ = ['blocking']


def blocking(clusters, options):
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
        analyzer = AnalyzeBlocking()
    else:
        analyzer = AnalyzeHeuristicBlocking(options)
    clusters = analyzer.process(clusters)
    clusters = AnalyzeSkewing().process(clusters)

    if options['blocklevels'] > 0:
        clusters = SynthesizeBlocking(options).process(clusters)

    if options['skewing']:
        clusters = SynthesizeSkewing(options).process(clusters)

    return clusters


class AnalyzeBlocking(Queue):

    """
    Encode the TILABLE property.
    """

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        for c in clusters:
            if not {PARALLEL, PARALLEL_IF_ATOMIC}.intersection(c.properties[d]):
                return clusters

        # All good, `d` is actually TILABLE
        processed = attach_property(clusters, d, TILABLE)

        return processed


class AnalyzeHeuristicBlocking(Queue):

    """
    Encode the TILABLE property based on heuristics.
    """

    def __init__(self, options):
        super().__init__()

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

        for c in clusters:
            # PARALLEL* and AFFINE are necessaary conditions
            if AFFINE not in c.properties[d] or \
               not ({PARALLEL, PARALLEL_IF_PVT} & c.properties[d]):
                return clusters

            # Heuristic: innermost Dimensions may be ruled out a-priori
            is_inner = d is c.itintervals[-1].dim
            if is_inner and not self.inner:
                return clusters

            # Heuristic: TILABLE not worth it if not within SEQUENTIAL Dimension
            if not any(SEQUENTIAL in c.properties[i.dim] for i in prefix[:-1]):
                return clusters

            # Heuristic: same as above if there's a local SubDimension
            if any(i.dim.is_Sub and i.dim.local for i in c.itintervals):
                return clusters

        if len(clusters) > 1:
            # Heuristic: same as above if it induces dynamic bounds
            exprs = flatten(c.exprs for c in as_tuple(clusters))
            scope = Scope(exprs)
            if any(i.is_lex_non_stmt for i in scope.d_all_gen()):
                return clusters
        else:
            # Just avoiding potentially expensive checks
            pass

        # All good, `d` is actually TILABLE
        processed = attach_property(clusters, d, TILABLE)

        return processed


class AnalyzeSkewing(Queue):

    """
    Encode the SKEWABLE Dimensions.
    """

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        for c in clusters:
            if TILABLE not in c.properties[d]:
                return clusters

        processed = attach_property(clusters, d, SKEWABLE)

        return processed


class SynthesizeBlocking(Queue):

    template = "%s%d_blk%s"

    def __init__(self, options):
        self.inner = bool(options['blockinner'])
        self.levels = options['blocklevels']

        self.nblocked = Counter()

        super().__init__()

    def _make_key_hook(self, cluster, level):
        return (tuple(cluster.guards.get(i.dim) for i in cluster.itintervals[:level]),)

    def _process_fdta(self, clusters, level, prefix=None):
        # Truncate recursion in case of TILABLE, non-perfect sub-nests, as
        # it's an unsupported case
        if prefix:
            d = prefix[-1].dim
            test0 = any(TILABLE in c.properties[d] for c in clusters)
            test1 = len({c.itintervals[:level] for c in clusters}) > 1
            if test0 and test1:
                return self.callback(clusters, prefix)

        return super()._process_fdta(clusters, level, prefix)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        # Create the block Dimensions (in total `self.levels` Dimensions)
        name = self.template % (d.name, self.nblocked[d], '%d')

        bd = BlockDimension(name % 0, d, d.symbolic_min, d.symbolic_max)
        size = bd.step
        block_dims = [bd]

        for i in range(1, self.levels):
            bd = BlockDimension(name % i, bd, bd, bd + bd.step - 1, size=size)
            block_dims.append(bd)

        bd = BlockDimension(d.name, bd, bd, bd + bd.step - 1, 1, size=size)
        block_dims.append(bd)

        processed = []
        for c in clusters:
            if TILABLE in c.properties[d]:
                ispace = decompose(c.ispace, d, block_dims)

                # Use the innermost BlockDimension in place of `d`
                exprs = [uxreplace(e, {d: bd}) for e in c.exprs]

                # The new Cluster properties
                # TILABLE property is dropped after the blocking.
                properties = dict(c.properties)
                properties.pop(d)
                properties.update({bd: c.properties[d] - {TILABLE} for bd in block_dims})

                processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                           properties=properties))
            else:
                processed.append(c)

        # Make sure to use unique BlockDimensions
        self.nblocked[d] += int(any(TILABLE in c.properties[d] for c in clusters))

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

    # Create the relations.
    # Example: consider the relation `(t, x, y)` and assume we decompose `x` over
    # `xbb, xb, xi`; then we decompose the relation as two relations, `(t, xbb, y)`
    # and `(xbb, xb, xi)`
    relations = [block_dims]
    for r in ispace.intervals.relations:
        relations.append([block_dims[0] if i is d else i for i in r])

    # Add more relations
    for n, i in enumerate(ispace):
        if i.dim is d:
            continue
        elif i.dim.is_Block:
            # Make sure BlockDimensions on the same level stick next to each other.
            # For example, we want `(t, xbb, ybb, xb, yb, x, y)`, rather than say
            # `(t, xbb, xb, x, ybb, ...)`
            for bd in block_dims:
                if i.dim._depth >= bd._depth:
                    relations.append([bd, i.dim])
                else:
                    relations.append([i.dim, bd])
        elif n > ispace.intervals.index(d):
            # The non-Block subsequent Dimensions must follow the block Dimensions
            for bd in block_dims:
                relations.append([bd, i.dim])
        else:
            # All other Dimensions must precede the block Dimensions
            for bd in block_dims:
                relations.append([i.dim, bd])

    intervals = IntervalGroup(intervals, relations=relations)

    sub_iterators = dict(ispace.sub_iterators)
    sub_iterators.pop(d, None)
    sub_iterators.update({bd: ispace.sub_iterators.get(d, []) for bd in block_dims})

    directions = dict(ispace.directions)
    directions.pop(d)
    directions.update({bd: ispace.directions[d] for bd in block_dims})

    return IterationSpace(intervals, sub_iterators, directions)


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

            # Since we are here, prefix is skewable and nested under a
            # SEQUENTIAL loop.
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
            processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                       properties=c.properties))

        return processed


# Utils


def attach_property(clusters, d, p):
    """
    Attach `p` to the `clusters`' properties.
    """
    processed = []
    for c in clusters:
        properties = dict(c.properties)
        properties[d] = set(properties[d]) | {p}
        processed.append(c.rebuild(properties=properties))

    return processed

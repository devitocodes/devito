from collections import Counter

from devito.ir.clusters import Queue
from devito.ir.support import TILABLE, IntervalGroup, IterationSpace
from devito.symbolics import uxreplace
from devito.tools import timed_pass
from devito.types import IncrDimension

from devito.ir.support import SEQUENTIAL, PARALLEL, Interval
from devito.logger import warning
from devito.symbolics import xreplace_indices

__all__ = ['Blocking', 'Skewing']


class Blocking(Queue):

    template = "%s%d_blk%s"

    def __init__(self, options):
        self.inner = bool(options['blockinner'])
        self.levels = options['blocklevels']

        self.nblocked = Counter()

        super(Blocking, self).__init__()

    def _make_key_hook(self, cluster, level):
        return (tuple(cluster.guards.get(i.dim) for i in cluster.itintervals[:level]),)

    @timed_pass(name='blocking')
    def process(self, clusters):
        # Preprocess: heuristic: drop TILABLE from innermost Dimensions to
        # maximize vectorization
        processed = []
        for c in clusters:
            ntilable = len([i for i in c.properties.values() if TILABLE in i])
            ntilable -= int(not self.inner)
            if ntilable <= 1:
                properties = {k: v - {TILABLE} for k, v in c.properties.items()}
                processed.append(c.rebuild(properties=properties))
            elif not self.inner:
                d = c.itintervals[-1].dim
                properties = dict(c.properties)
                properties[d] = properties[d] - {TILABLE}
                processed.append(c.rebuild(properties=properties))
            else:
                processed.append(c)

        processed = super(Blocking, self).process(processed)

        return processed

    def _process_fdta(self, clusters, level, prefix=None):
        # Truncate recursion in case of TILABLE, non-perfect sub-nests, as
        # it's an unsupported case
        if prefix:
            d = prefix[-1].dim
            test0 = any(TILABLE in c.properties[d] for c in clusters)
            test1 = len({c.itintervals[:level] for c in clusters}) > 1
            if test0 and test1:
                return self.callback(clusters, prefix)

        return super(Blocking, self)._process_fdta(clusters, level, prefix)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        # Create the block Dimensions (in total `self.levels` Dimensions)
        name = self.template % (d.name, self.nblocked[d], '%d')

        bd = IncrDimension(name % 0, d, d.symbolic_min, d.symbolic_max)
        size = bd.step
        block_dims = [bd]

        for i in range(1, self.levels):
            bd = IncrDimension(name % i, bd, bd, bd + bd.step - 1, size=size)
            block_dims.append(bd)

        bd = IncrDimension(d.name, bd, bd, bd + bd.step - 1, 1, size=size)
        block_dims.append(bd)

        processed = []
        for c in clusters:
            if TILABLE in c.properties[d]:
                ispace = decompose(c.ispace, d, block_dims)

                # Use the innermost IncrDimension in place of `d`
                exprs = [uxreplace(e, {d: bd}) for e in c.exprs]

                # The new Cluster properties
                properties = dict(c.properties)
                properties.pop(d)
                properties.update({bd: c.properties[d] - {TILABLE} for bd in block_dims})

                processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                           properties=properties))
            else:
                processed.append(c)

        # Make sure to use unique IncrDimensions
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

    # The level of a given Dimension in the hierarchy of block Dimensions
    level = lambda dim: len([i for i in dim._defines if i.is_Incr])

    # Add more relations
    for n, i in enumerate(ispace):
        if i.dim is d:
            continue
        elif i.dim.is_Incr:
            # Make sure IncrDimensions on the same level stick next to each other.
            # For example, we want `(t, xbb, ybb, xb, yb, x, y)`, rather than say
            # `(t, xbb, xb, x, ybb, ...)`
            for bd in block_dims:
                if level(i.dim) >= level(bd):
                    relations.append([bd, i.dim])
                else:
                    relations.append([i.dim, bd])
        elif n > ispace.intervals.index(d):
            # The non-Incr subsequent Dimensions must follow the block Dimensions
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


class Skewing(Queue):

    def __init__(self, options):

        self.nskewed = Counter()

        super(Skewing, self).__init__()

    @timed_pass(name='skewing')
    def process(self, clusters):
        processed = super(Skewing, self).process(clusters)

        return processed

    def _process_fdta(self, clusters, level, prefix=None):

        return super(Skewing, self)._process_fdta(clusters, level, prefix)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        processed = []
        for c in clusters:
            if PARALLEL in c.properties[d] and not d.symbolic_incr.is_Symbol:
                ispace, exprs = skew(c.ispace, c.exprs, c.properties, d)
                processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                           properties=c.properties))
            else:
                processed.append(c)

        return processed


def skew(ispace, exprs, properties, d):

    """
    Create a new IterationSpace and skew expressions
    Skew the accesses along a SEQUENTIAL Dimension.
    Reference:
    https://link.springer.com/article/10.1007%2FBF01407876
    Example:

    Transform

    for i = 2, n-1
        for j = 2, m-1
            a[i,j] = (a[a-1,j] + a[i,j-1] + a[i+1,j] + a[i,j+1]) / 4
        end for
    end for

    to

    for i = 2, n-1
        for j = 2+i, m-1+i
            a[i,j-i] = (a[a-1,j-i] + a[i,j-1-i] + a[i+1,j-i] + a[i,j+1-i]) / 4
        end for
    end for
    """

    # What dimensions should we skew against?
    # e.g. SEQUENTIAL Dimensions (Usually time in FD solvers, but not only limited
    # to this)
    # What dimensions should be skewed?
    # e.g. PARALLEL dimensions (x, y, z) but not blocking ones (x_blk0, y_blk0)
    # Iterate over the iteration space and assign dimensions to skew or skewable
    # depending on their properties

    skew_dims, skewable = [], []
    skewable.append(d)

    for dim in ispace.intervals.dimensions:
        if SEQUENTIAL in properties[dim] and not dim.symbolic_incr.is_Symbol:
            skew_dims.append(dim)

    if len(skew_dims) > 1:
        raise warning("More than 1 dimensions that can be skewed.\
                    Skewing the first in the list")
    elif len(skew_dims) == 0:
        # No dimensions to skew against -> nothing to do, return
        return ispace, exprs

    # Skew dim will not be none here:
    # Initializing a default skewed dim index position in loop
    skew_dim = skew_dims.pop()  # Skew first one

    mapper, intervals, processed = {}, [], []

    for i in ispace.intervals:
        # Skew a dim if nested under skew_dim and is prefix:
        if ispace.intervals.index(skew_dim) < ispace.intervals.index(i) and i.dim == d:
            mapper[i.dim] = i.dim - skew_dim
            intervals.append(Interval(i.dim, skew_dim, skew_dim))
        # Do not touch otherwise
        else:
            intervals.append(i)

        processed = xreplace_indices(exprs, mapper)

    ispace = IterationSpace(intervals, ispace.sub_iterators,
                            ispace.directions)

    return ispace, processed

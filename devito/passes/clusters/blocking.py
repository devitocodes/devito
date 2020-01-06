from collections import Counter

import numpy as np
from cached_property import cached_property

from devito.ir.clusters import Cluster, Queue
from devito.ir.support import TILABLE, IntervalGroup, IterationSpace
from devito.tools import timed_pass
from devito.types import IncrDimension, Scalar

__all__ = ['Blocking', 'IncrDimension']


class Blocking(Queue):

    template = "%s%d_blk%s"

    def __init__(self, inner, levels):
        self.inner = bool(inner)
        self.levels = levels

        self.nblocked = Counter()

        super(Blocking, self).__init__()

    @timed_pass(name='specializing.Clusters.blocking')
    def process(self, elements):
        return super(Blocking, self).process(elements)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        processed = []
        for c in clusters:
            if TILABLE not in c.properties[d]:
                processed.append(c)
            elif self.inner is False and (d is c.itintervals[-1].dim):
                # By default, the innermost Dimension isn't tiled. This behaviour
                # may have been overwritten by the user, in which case `self.inner`
                # will be False
                processed.append(c)
            else:
                processed.append(self._callback(c, d))
                self.nblocked[d] += 1

        return processed

    def _callback(self, cluster, d):
        # Actually apply blocking

        name = self.template % (d.name, self.nblocked[d], '%d')

        # Create the block Dimensions (in total `self.levels` Dimensions)

        bd = IncrDimension(d, d.symbolic_min, d.symbolic_max, name=name % 0)
        block_dims = [bd]

        for i in range(1, self.levels):
            bd = IncrDimension(bd, bd, bd + bd.step - 1, name=name % i)
            block_dims.append(bd)

        bd = IncrDimension(bd, bd, bd + bd.step - 1, 1, d.name)
        block_dims.append(bd)

        # Exploit the newly created IncrDimensions in the new IterationSpace
        ispace = decompose(cluster.ispace, d, block_dims)

        # Use the innermost BlockDimension in place of `d` within the expressions
        exprs = [e.xreplace({d: bd}) for e in cluster.exprs]

        # The new Cluster properties
        properties = dict(cluster.properties)
        properties.pop(d)
        properties.update({bd: cluster.properties[d] - {TILABLE} for bd in block_dims})

        return cluster.rebuild(exprs=exprs, ispace=ispace, properties=properties)


def decompose(ispace, d, block_dims):
    """
    Create a new IterationSpace in which the `d` Interval is decomposed
    into a hierarchy of Intervals over ``block_dims``.
    """
    # Create the new Intervals
    intervals = []
    for i in ispace.intervals:
        if i.dim is d:
            intervals.append(i.switch(block_dims[0]))
            intervals.extend([i.switch(bd).zero() for bd in block_dims[1:]])
        else:
            intervals.append(i)

    # Create the new "decomposed" relations.
    # Example: consider the relation `(t, x, y)` and assume we decompose `x` over
    # `xbb, xb, xi`; then we decompose the relation as two relations, `(t, xbb, y)`
    # and `(xbb, xb, xi)`
    relations = [block_dims]
    for r in ispace.intervals.relations:
        relations.append([block_dims[0] if i is d else i for i in r])

    # Further, if there are other IncrDimensions, add relations such that
    # IncrDimensions at the same level stick together, thus we obtain for
    # example `(t, xbb, ybb, xb, yb, x, y)` instead of `(t, xbb, xb, x, ybb, ...)`
    for i in intervals:
        if not isinstance(i.dim, IncrDimension):
            continue
        for bd in block_dims:
            if bd._defines & i.dim._defines:
                break
            if len(i.dim._defines) > len(bd._defines):
                relations.append([bd, i.dim])

    intervals = IntervalGroup(intervals, relations=relations)

    sub_iterators = dict(ispace.sub_iterators)
    sub_iterators.pop(d, None)
    sub_iterators.update({bd: ispace.sub_iterators.get(d, []) for bd in block_dims})

    directions = dict(ispace.directions)
    directions.pop(d)
    directions.update({bd: ispace.directions[d] for bd in block_dims})

    return IterationSpace(intervals, sub_iterators, directions)

from collections import Counter

import numpy as np
from cached_property import cached_property

from devito.ir.clusters import Cluster, ClusterCompound, Queue
from devito.ir.support import TILABLE
from devito.types import IncrDimension, Scalar

__all__ = ['Blocking', 'BlockDimension']


class Blocking(Queue):

    template = "%s%d_blk%s"

    def __init__(self, inner, levels):
        self.inner = bool(inner)
        self.levels = levels

        self.nblocked = Counter()

        super(Blocking, self).__init__()

    def process(self, elements):
        #TODO: necessary??
        return self._process_fatd(elements, 1)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        processed = []
        for c in clusters:
            if TILABLE in c.properties.get(d, []):
                processed.append(self._callback(c, d))
                self.nblocked[d] += 1
            else:
                processed.append(c)

        return processed

    def _callback(self, cluster, d):
        # Actually apply blocking

        name = self.template % (d.name, self.nblocked[d], '%d')

        # Create the BlockDimensions
        bd = d
        dims = []
        for i in range(self.levels):
            bd = BlockDimension(bd, name=name % i)
            dims.append(bd)
        dims.append(BlockDimension(bd, name=d.name, step=1))

        # Create a new IterationSpace using the BlockDimensions
        ispace = cluster.ispace.decompose(d, dims)

        # Update the Cluster properties
        properties = dict(cluster.properties)
        for i in dims:
            properties[i] = properties[d] - {TILABLE}
        properties.pop(d)

        #TODO: DataSpace ??

        return cluster.rebuild(ispace=ispace, properties=properties)


class BlockDimension(IncrDimension):

    is_PerfKnob = True

    @cached_property
    def symbolic_min(self):
        return Scalar(name=self.min_name, dtype=np.int32, is_const=True)

    @property
    def _arg_names(self):
        return (self.step.name,)

    def _arg_defaults(self, **kwargs):
        # TODO: need a heuristic to pick a default block size
        return {self.step.name: 8}

    def _arg_values(self, args, interval, grid, **kwargs):
        if self.step.name in kwargs:
            return {self.step.name: kwargs.pop(self.step.name)}
        elif isinstance(self.parent, BlockDimension):
            # `self` is a BlockDimension within an outer BlockDimension, but
            # no value supplied -> the sub-block will span the entire block
            return {self.step.name: args[self.parent.step.name]}
        else:
            value = self._arg_defaults()[self.step.name]
            if value <= args[self.root.max_name] - args[self.root.min_name] + 1:
                return {self.step.name: value}
            else:
                # Avoid OOB (will end up here only in case of tiny iteration spaces)
                return {self.step.name: 1}

    def _arg_check(self, args, interval):
        """Check the block size won't cause OOB accesses."""
        value = args[self.step.name]
        if isinstance(self.parent, BlockDimension):
            # sub-BlockDimensions must be perfect divisors of their parent
            parent_value = args[self.parent.step.name]
            if parent_value % value > 0:
                raise InvalidArgument("Illegal block size `%s=%d`: sub-block sizes "
                                      "must divide the parent block size evenly (`%s=%d`)"
                                      % (self.step.name, value,
                                         self.parent.step.name, parent_value))
        else:
            if value < 0:
                raise InvalidArgument("Illegal block size `%s=%d`: it should be > 0"
                                      % (self.step.name, value))
            if value > args[self.root.max_name] - args[self.root.min_name] + 1:
                # Avoid OOB
                raise InvalidArgument("Illegal block size `%s=%d`: it's greater than the "
                                      "iteration range and it will cause an OOB access"
                                      % (self.step.name, value))

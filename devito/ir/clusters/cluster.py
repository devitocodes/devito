from collections import defaultdict

from cached_property import cached_property

from devito.ir.dfg import TemporariesGraph
from devito.tools import as_tuple

__all__ = ["Cluster", "ClusterGroup"]


class PartialCluster(object):

    """
    A PartialCluster is an ordered sequence of expressions that result in the
    computation of a tensor, plus the tensor expression itself.

    PartialClusters are mutable -- in particular, their stencil as well as
    the embedded sequence of expressions are subjected to modifications.
    """

    def __init__(self, exprs, stencil):
        """
        Initialize a PartialCluster.

        :param exprs: The ordered sequence of expressions computing a tensor.
        :param stencil: An object of type :class:`Stencil`, which provides
                        information about the points accessed along each dimension
                        to compute the output tensor.
        """
        self._exprs = list(exprs)
        self._stencil = stencil

    @property
    def exprs(self):
        return self._exprs

    @property
    def stencil(self):
        return self._stencil

    @property
    def trace(self):
        return TemporariesGraph(self.exprs)

    @property
    def unknown(self):
        return self.trace.unknown

    @property
    def tensors(self):
        return self.trace.tensors

    @exprs.setter
    def exprs(self, val):
        self._exprs = val

    @stencil.setter
    def stencil(self, val):
        self._stencil = val

    def squash(self, other):
        """Concatenate the expressions in ``other`` to those in ``self``.
        ``self`` and ``other`` must have same ``stencil``. Duplicate
        expressions are dropped."""
        assert self.stencil == other.stencil
        self.exprs.extend([i for i in other.exprs if i not in self.exprs])


class Cluster(PartialCluster):

    """A Cluster is an immutable PartialCluster."""

    def __init__(self, exprs, stencil):
        self._exprs = as_tuple(exprs)
        self._stencil = stencil.frozen

    @cached_property
    def trace(self):
        return TemporariesGraph(self.exprs)

    @property
    def is_dense(self):
        return self.trace.space_indices and not self.trace.time_invariant()

    @property
    def is_sparse(self):
        return not self.is_dense

    def rebuild(self, exprs):
        """
        Build a new cluster with expressions ``exprs`` having same stencil as ``self``.
        """
        return Cluster(exprs, self.stencil)

    @PartialCluster.exprs.setter
    def exprs(self, val):
        raise AttributeError

    @PartialCluster.stencil.setter
    def stencil(self, val):
        raise AttributeError

    def squash(self, other):
        raise AttributeError


class ClusterGroup(list):

    def __init__(self, *clusters):
        """
        An iterable of Clusters, which also tracks the atomic dimensions
        of each of its elements; that is, those dimensions that a Cluster
        cannot share with an adjacent cluster, to honor data dependences.
        """
        super(ClusterGroup, self).__init__(*clusters)
        self.atomics = defaultdict(set)

    def unfreeze(self):
        """
        Return a new ClusterGroup in which all of ``self``'s Clusters have
        been promoted to PartialClusters. The ``atomics`` information is lost.
        """
        return ClusterGroup([PartialCluster(i.exprs, i.stencil)
                             if isinstance(i, Cluster) else i for i in self])

    def freeze(self):
        """
        Return a new ClusterGroup in which all of ``self``'s PartialClusters
        have been turned into actual Clusters.
        """
        clusters = ClusterGroup()
        for i in self:
            if isinstance(i, PartialCluster):
                cluster = Cluster(i.exprs, i.stencil)
                clusters.append(cluster)
                clusters.atomics[cluster] = self.atomics[i]
            else:
                clusters.append(i)
        return clusters

from collections import defaultdict

from sympy import Eq
from cached_property import cached_property

from devito.ir.support import Box
from devito.ir.clusters.graph import FlowGraph

__all__ = ["Cluster", "ClusterGroup"]


class PartialCluster(object):

    """
    A PartialCluster is an ordered sequence of scalar expressions that contribute
    to the computation of a tensor, plus the tensor expression itself.

    A PartialCluster is mutable.

    :param exprs: The ordered sequence of expressions computing a tensor.
    :param ispace: An object of type :class:`Box`, representing the iteration
                   space of the cluster.
    """

    def __init__(self, exprs, ispace):
        self._exprs = list(exprs)
        self._ispace = ispace

    @property
    def exprs(self):
        return self._exprs

    @property
    def ispace(self):
        return self._ispace

    @property
    def trace(self):
        return FlowGraph(self.exprs)

    @property
    def unknown(self):
        return self.trace.unknown

    @property
    def tensors(self):
        return self.trace.tensors

    @exprs.setter
    def exprs(self, val):
        self._exprs = val

    @ispace.setter
    def ispace(self, val):
        raise AttributeError

    def squash(self, other):
        """Concatenate the expressions in ``other`` to those in ``self``.
        ``self`` and ``other`` must have same ``ispace``. Duplicate
        expressions are dropped."""
        assert self.ispace == other.ispace
        self.exprs.extend([i for i in other.exprs if i not in self.exprs])


class Cluster(PartialCluster):

    """A Cluster is an immutable :class:`PartialCluster`."""

    def __init__(self, exprs, ispace):
        self._exprs = tuple(Eq(*i.args, evaluate=False) for i in exprs)
        self._ispace = ispace

    @cached_property
    def trace(self):
        return FlowGraph(self.exprs)

    @property
    def is_dense(self):
        return self.trace.space_indices and not self.trace.time_invariant()

    @property
    def is_sparse(self):
        return not self.is_dense

    def rebuild(self, exprs):
        """
        Build a new cluster with expressions ``exprs`` having same iteration
        space as ``self``.
        """
        return Cluster(exprs, self.ispace)

    @PartialCluster.exprs.setter
    def exprs(self, val):
        raise AttributeError

    def squash(self, other):
        raise AttributeError


class ClusterGroup(list):

    """
    An iterable of Clusters, which also tracks the atomic dimensions
    of each of its elements; that is, those dimensions that a Cluster
    cannot share with an adjacent cluster, to honor data dependences.
    """

    def __init__(self, *clusters):
        super(ClusterGroup, self).__init__(*clusters)
        self.atomics = defaultdict(set)

    @property
    def ispace(self):
        """
        Return the union of all Clusters' iteration spaces.
        """
        if not self:
            return Box([])
        return Box.intersection_update(*[i.ispace for i in self])

    def unfreeze(self):
        """
        Return a new ClusterGroup in which all of ``self``'s Clusters have
        been promoted to PartialClusters. The ``atomics`` information is lost.
        """
        return ClusterGroup([PartialCluster(i.exprs, i.ispace)
                             if isinstance(i, Cluster) else i for i in self])

    def finalize(self):
        """
        Return a new ClusterGroup in which all of ``self``'s PartialClusters
        have been turned into actual Clusters.
        """
        clusters = ClusterGroup()
        for i in self:
            if isinstance(i, PartialCluster):
                cluster = Cluster(i.exprs, i.ispace)
                clusters.append(cluster)
                clusters.atomics[cluster] = self.atomics[i]
            else:
                clusters.append(i)
        return clusters

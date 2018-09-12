import numpy as np
from cached_property import cached_property
from frozendict import frozendict

from devito.ir.equations import ClusterizedEq
from devito.ir.clusters.graph import FlowGraph
from devito.ir.support import DataSpace, IterationSpace, detect_io
from devito.symbolics import estimate_cost
from devito.tools import as_tuple

__all__ = ["Cluster", "ClusterGroup"]


class PartialCluster(object):

    """
    A PartialCluster is an ordered sequence of scalar expressions that contribute
    to the computation of a tensor, plus the tensor expression itself.

    A PartialCluster is mutable.

    :param exprs: The ordered sequence of expressions computing a tensor.
    :param ispace: An object of type :class:`IterationSpace`, which represents the
                   iteration space of the cluster.
    :param dspace: An object of type :class:`DataSpace`, which represents the
                   data space (i.e., data items accessed) of the cluster.
    :param atomics: (Optional) non-sharable :class:`Dimension`s in ``ispace``.
    :param guards: (Optional) iterable of conditions, provided as SymPy expressions,
                   under which ``exprs`` are evaluated.
    """

    def __init__(self, exprs, ispace, dspace, atomics=None, guards=None):
        self._exprs = list(ClusterizedEq(i, ispace=ispace, dspace=dspace)
                           for i in as_tuple(exprs))
        self._ispace = ispace
        self._dspace = dspace
        self._atomics = set(atomics or [])
        self._guards = guards or {}

    @property
    def exprs(self):
        return self._exprs

    @property
    def ispace(self):
        return self._ispace

    @property
    def itintervals(self):
        return self._ispace.itintervals

    @property
    def extent(self):
        return self.ispace.extent

    @property
    def shape(self):
        return self.ispace.shape

    @property
    def dspace(self):
        return self._dspace

    @property
    def atomics(self):
        return self._atomics

    @property
    def guards(self):
        return self._guards

    @property
    def args(self):
        return (self.exprs, self.ispace, self.dspace, self.atomics, self.guards)

    @property
    def trace(self):
        return FlowGraph(self.exprs)

    @property
    def unknown(self):
        return self.trace.unknown

    @property
    def tensors(self):
        return self.trace.tensors

    @property
    def dtype(self):
        """Return the arithmetic data type of this Cluster. If the Cluster is
        performing floating point arithmetic, then any equation performing
        integer arithmetic is ignored, assuming that they are only carrying
        out array index calculations. If two equations are performing floating
        point calculations with mixed precision, return the data type with
        highest precision."""
        dtypes = {i.dtype for i in self.exprs}
        fdtypes = {i for i in dtypes if np.issubdtype(i, np.floating)}
        if len(fdtypes) > 1:
            raise NotImplementedError("Unsupported Cluster with mixed floating "
                                      "point arithmetic %s" % str(fdtypes))
        elif len(fdtypes) == 1:
            return fdtypes.pop()
        elif len(dtypes) == 1:
            return dtypes.pop()
        else:
            raise ValueError("Unsupported Cluster [mixed integer arithmetic ?]")

    @property
    def ops(self):
        """
        Return the floating point operations performed by this Cluster.
        """
        return self.extent*sum(estimate_cost(i) for i in self.exprs)

    @property
    def traffic(self):
        """
        Return the compulsary traffic (number of reads/writes) generated
        by this Cluster, as a mapper from tensor objects to :class:`IntervalGroup`s.

        .. note::

            If a tensor object is both read and written, then it is counted twice.
        """
        reads, writes = detect_io(self.exprs, relax=True)
        accesses = [(i, 'r') for i in reads] + [(i, 'w') for i in writes]
        ret = {}
        for i, mode in accesses:
            if not i.is_Tensor:
                continue
            elif i in self.dspace.parts:
                # Stencils extend the data spaces beyond the iteration spaces
                intervals = self.dspace.parts[i]
                # Assume that invariant dimensions always cause new loads/stores
                invariants = self.ispace.intervals.drop(intervals.dimensions)
                intervals = intervals.generate('union', invariants, intervals)
                ret[(i, mode)] = intervals
            else:
                ret[(i, mode)] = self.ispace.intervals
        return ret

    @exprs.setter
    def exprs(self, val):
        self._exprs = val

    @ispace.setter
    def ispace(self, val):
        self._ispace = val

    @dspace.setter
    def dspace(self, val):
        self._dspace = val

    def squash(self, other):
        """Concatenate the expressions in ``other`` to those in ``self``.
        ``self`` and ``other`` must have same ``ispace``. Duplicate
        expressions are dropped. The :class:`DataSpace` is updated
        accordingly."""
        assert self.ispace.is_compatible(other.ispace)
        self.exprs.extend([i for i in other.exprs if i not in self.exprs])
        self.dspace = DataSpace.merge(self.dspace, other.dspace)
        self.ispace = IterationSpace.merge(self.ispace, other.ispace)


class Cluster(PartialCluster):

    """A Cluster is an immutable :class:`PartialCluster`."""

    def __init__(self, exprs, ispace, dspace, atomics=None, guards=None):
        self._exprs = exprs
        # Keep expressions ordered based on information flow
        self._exprs = tuple(ClusterizedEq(v, ispace=ispace, dspace=dspace)
                            for v in self.trace.values())
        self._ispace = ispace
        self._dspace = dspace
        self._atomics = frozenset(atomics or ())
        self._guards = frozendict(guards or {})

    @cached_property
    def trace(self):
        return FlowGraph(self.exprs)

    @cached_property
    def functions(self):
        return set.union(*[set(i.dspace.parts) for i in self.exprs])

    @cached_property
    def is_sparse(self):
        return any(f.is_SparseFunction for f in self.functions)

    @cached_property
    def is_dense(self):
        return not self.is_sparse

    def rebuild(self, exprs):
        """
        Build a new cluster with expressions ``exprs`` having same iteration
        space and atomics as ``self``.
        """
        return Cluster(exprs, self.ispace, self.dspace, self.atomics, self.guards)

    @PartialCluster.exprs.setter
    def exprs(self, val):
        raise AttributeError

    @PartialCluster.ispace.setter
    def ispace(self, val):
        raise AttributeError

    @PartialCluster.dspace.setter
    def dspace(self, val):
        raise AttributeError

    def squash(self, other):
        raise AttributeError


class ClusterGroup(list):

    """An iterable of :class:`PartialCluster`s."""

    def unfreeze(self):
        """
        Return a new ClusterGroup in which all of ``self``'s Clusters have
        been promoted to PartialClusters. Any metadata attached to self is lost.
        """
        return ClusterGroup([PartialCluster(*i.args) if isinstance(i, Cluster) else i
                             for i in self])

    def finalize(self):
        """
        Return a new ClusterGroup in which all of ``self``'s PartialClusters
        have been turned into actual Clusters.
        """
        clusters = ClusterGroup()
        for i in self:
            if isinstance(i, PartialCluster):
                cluster = Cluster(*i.args)
                clusters.append(cluster)
            else:
                clusters.append(i)
        return clusters

    @property
    def dspace(self):
        """Return the cumulative :class:`DataSpace` of this ClusterGroup."""
        return DataSpace.merge(*[i.dspace for i in self])

    @property
    def dtype(self):
        """Return the arithmetic data type of this ClusterGroup. If at least one
        Cluster is performing floating point arithmetic, then any Cluster performing
        integer arithmetic is ignored. If two Clusters are performing floating
        point calculations with different precision, return the data type with
        highest precision."""
        dtypes = {i.dtype for i in self}
        fdtypes = {i for i in dtypes if np.issubdtype(i, np.floating)}
        if len(fdtypes) > 1:
            raise NotImplementedError("Unsupported ClusterGroup with mixed floating "
                                      "point arithmetic %s" % str(fdtypes))
        elif len(fdtypes) == 1:
            return fdtypes.pop()
        elif len(dtypes) == 1:
            return dtypes.pop()
        else:
            raise ValueError("Unsupported ClusterGroup [mixed integer arithmetic ?]")

    @property
    def meta(self):
        """Return the metadata carried by this ClusterGroup, a 2-tuple consisting
        of data type and data space."""
        return (self.dtype, self.dspace)

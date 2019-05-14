import numpy as np
from cached_property import cached_property
from frozendict import frozendict

from devito.ir.equations import ClusterizedEq
from devito.ir.clusters.graph import FlowGraph
from devito.ir.support import DataSpace, IterationSpace, detect_io
from devito.symbolics import estimate_cost
from devito.tools import as_tuple

__all__ = ["Cluster", "ClusterGroup"]

# Handling of skewed loops to be added
<<<<<<< HEAD
=======

>>>>>>> Init Sims diff
class PartialCluster(object):

    """
    A PartialCluster is an ordered sequence of scalar expressions contributing
    to the computation of a tensor, plus the tensor expression itself.

    A PartialCluster is mutable.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        An ordered sequence of expressions computing a tensor.
    ispace : IterationSpace
        The cluster iteration space.
    dspace : DataSpace
        The cluster data space.
    atomics : list, optional
        Dimensions inducing a data dependence with other PartialClusters.
    guards : dict
        Mapper from Dimensions to expr-like, representing the conditions under
        which the PartialCluster should be computed.
    """

    def __init__(self, exprs, ispace, dspace, atomics=None, guards=None, skewed_loops={}):
        self._exprs = list(ClusterizedEq(i, ispace=ispace, dspace=dspace)
                           for i in as_tuple(exprs))
        #, skewed_loops={}This causes hanging on tests until now.To do it with caution...
        self._ispace = ispace
        self._dspace = dspace
        self._atomics = set(atomics or [])
        self._guards = guards or {}
        self._skewed_loops = skewed_loops

    @property
    def exprs(self):
        return self._exprs

    @property
    def ispace(self):
        return self._ispace

    @property
    def dimensions(self):
        return self._ispace.dimensions

    @property
    def itintervals(self):
        return self._ispace.itintervals

    @property
    def size(self):
        return self.ispace.size

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
    def skewed_loops(self):
        return self._skewed_loops

    @property
    def args(self):
        return (self.exprs, self.ispace, self.dspace, self.atomics, self.guards)

    @property
    def flowgraph(self):
        return FlowGraph(self.exprs)

    @property
    def unknown(self):
        return self.flowgraph.unknown

    @property
    def tensors(self):
        return self.flowgraph.tensors

    @property
    def dtype(self):
        """
        The arithmetic data type of the . If the Cluster performs
        floating point arithmetic, then the expressions performing integer
        arithmetic are ignored, assuming that they are only carrying out array
        index calculations. If two expressions perform floating point
        calculations with mixed precision, the data type with highest precision
        is returned.
        """
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
        """The Cluster operation count."""
        return self.size*sum(estimate_cost(i) for i in self.exprs)

    @property
    def traffic(self):
        """
        The Cluster compulsary traffic (number of reads/writes), as a mapper
        from Functions to IntervalGroups.

        Notes
        -----
        If a Function is both read and written, then it is counted twice.
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

    @skewed_loops.setter
    def skewed_loops(self, val):
        self._skewed_loops = val

    def squash(self, other):
        """
        Concatenate the expressions in ``other`` to those in ``self``.
        ``self`` and ``other`` must have same ``ispace``. Duplicate expressions
        are dropped. The DataSpace is updated accordingly.
        """
        assert self.ispace.is_compatible(other.ispace)
        self.exprs.extend([i for i in other.exprs
                           if i not in self.exprs or i.is_Increment])
        self.dspace = DataSpace.merge(self.dspace, other.dspace)
        self.ispace = IterationSpace.merge(self.ispace, other.ispace)


class Cluster(PartialCluster):

    """A Cluster is an immutable PartialCluster."""

    def __init__(self, exprs, ispace, dspace, atomics=None, guards=None, skewed_loops={}):
        self._exprs = exprs
        # Keep expressions ordered based on information flow
        self._exprs = tuple(ClusterizedEq(v, ispace=ispace, dspace=dspace)
                            for v in self.flowgraph.values())
        self._ispace = ispace
        self._dspace = dspace
        self._atomics = frozenset(atomics or ())
        self._guards = frozendict(guards or {})
        self._skewed_loops = skewed_loops

    @cached_property
    def flowgraph(self):
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
        Build a new Cluster with expressions ``exprs`` having same iteration
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

    @PartialCluster.skewed_loops.setter
    def skewed_loops(self, val):
        raise AttributeError

    def squash(self, other):
        raise AttributeError


class ClusterGroup(list):

    """An iterable of PartialClusters."""

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
        """Return the DataSpace of this ClusterGroup."""
        return DataSpace.merge(*[i.dspace for i in self])

    @property
    def dtype(self):
        """
        The arithmetic data type of this ClusterGroup. If at least one
        Cluster performs floating point arithmetic, then Clusters performing
        integer arithmetic are ignored. If two Clusters perform floating
        point calculations with different precision, return the data type with
        highest precision.
        """
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
        """
        Returns
        -------
        dtype, DSpace
            The data type and the data space of the ClusterGroup.
        """
        return (self.dtype, self.dspace)

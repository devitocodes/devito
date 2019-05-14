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
    guards : dict, optional
        Mapper from Dimensions to expr-like, representing the conditions under
        which the PartialCluster should be computed.
    local : list, optional
        Write-once/read-once Arrays. The write is performed by this PartialCluster,
        while the read by another PartialCluster.
    """

    def __init__(self, exprs, ispace, dspace, atomics=None, guards=None, local=None):
        self._exprs = list(ClusterizedEq(i, ispace=ispace, dspace=dspace)
                           for i in as_tuple(exprs))
        self._ispace = ispace
        self._dspace = dspace
        self._atomics = set(atomics or [])
        self._guards = guards or {}
        self._locals = set(local or [])

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
    def locals(self):
        return self._locals

    @property
    def args(self):
        return (self.exprs, self.ispace, self.dspace, self.atomics, self.guards,
                self.locals)

    @property
    def flowgraph(self):
        return FlowGraph(self.exprs)

    @property
    def tensors(self):
        return self.flowgraph.tensors

    @property
    def dtype(self):
        """
        The arithmetic data type of the Cluster. If the Cluster performs
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

    def merge(self, other):
        """
        Concatenate the expressions in ``other`` to those in ``self``.
        ``self`` and ``other`` must have same ``ispace``. Duplicate expressions
        are dropped. The DataSpace as well as the other metadata are updated
        accordingly.
        """
        assert self.ispace.is_compatible(other.ispace)
        self._exprs.extend([i for i in other.exprs
                            if i not in self.exprs or i.is_Increment])
        self._ispace = IterationSpace.merge(self.ispace, other.ispace)
        self._dspace = DataSpace.merge(self.dspace, other.dspace)
        self._atomics.update(other.atomics)
        self._guards.update(other.guards)
        self._locals.update(other.locals)


class Cluster(PartialCluster):

    """A Cluster is an immutable PartialCluster."""

    def __init__(self, exprs, ispace, dspace, atomics=None, guards=None, local=None):
        self._exprs = exprs
        # Keep expressions ordered based on information flow
        self._exprs = tuple(ClusterizedEq(v, ispace=ispace, dspace=dspace)
                            for v in self.flowgraph.values())
        self._ispace = ispace
        self._dspace = dspace
        self._atomics = frozenset(atomics or [])
        self._guards = frozendict(guards or {})
        self._locals = frozenset(local or [])

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
        return Cluster(exprs, self.ispace, self.dspace, self.atomics,
                       self.guards, self.locals)

    def squash(self, other):
        raise AttributeError


class ClusterGroup(tuple):

    """An immutable iterable of Clusters."""

    def __new__(cls, items):
        items = [Cluster(*i.args) if isinstance(i, PartialCluster) else i
                 for i in items]
        assert all(isinstance(i, Cluster) for i in items)
        return super(ClusterGroup, cls).__new__(cls, items)

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

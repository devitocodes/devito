from itertools import chain

import numpy as np
from cached_property import cached_property

from devito.ir.equations import ClusterizedEq
from devito.ir.support import (PARALLEL, PARALLEL_IF_PVT, BaseGuardBoundNext,
                               Forward, Interval, IntervalGroup, IterationSpace,
                               DataSpace, Guards, Properties, Scope, detect_accesses,
                               detect_io, normalize_properties, normalize_syncs,
                               sdims_min, sdims_max)
from devito.symbolics import estimate_cost
from devito.tools import as_tuple, flatten, frozendict, infer_dtype

__all__ = ["Cluster", "ClusterGroup"]


class Cluster(object):

    """
    A Cluster is an ordered sequence of expressions in an IterationSpace.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        An ordered sequence of expressions computing a tensor.
    ispace : IterationSpace, optional
        The cluster iteration space.
    guards : dict, optional
        Mapper from Dimensions to expr-like, representing the conditions under
        which the Cluster should be computed.
    properties : dict, optional
        Mapper from Dimensions to Property, describing the Cluster properties
        such as its parallel Dimensions.
    syncs : dict, optional
        Mapper from Dimensions to lists of SyncOps, that is ordered sequences of
        synchronization operations that must be performed in order to compute the
        Cluster asynchronously.
    """

    def __init__(self, exprs, ispace=None, guards=None, properties=None, syncs=None):
        ispace = ispace or IterationSpace([])

        self._exprs = tuple(ClusterizedEq(e, ispace=ispace) for e in as_tuple(exprs))
        self._ispace = ispace
        self._guards = Guards(guards or {})
        self._syncs = frozendict(syncs or {})

        # Normalize properties
        properties = Properties(properties or {})
        for d in ispace.itdimensions:
            properties = properties.add(d)
        for i in properties:
            for d in as_tuple(i):
                if d not in ispace.itdimensions:
                    properties = properties.drop(d)
        self._properties = properties

    def __repr__(self):
        return "Cluster([%s])" % ('\n' + ' '*9).join('%s' % i for i in self.exprs)

    @classmethod
    def from_clusters(cls, *clusters):
        """
        Build a new Cluster from a sequence of pre-existing Clusters with
        compatible IterationSpace.
        """
        assert len(clusters) > 0
        root = clusters[0]
        if not all(root.ispace.is_compatible(c.ispace) for c in clusters):
            raise ValueError("Cannot build a Cluster from Clusters with "
                             "incompatible IterationSpace")
        if not all(root.guards == c.guards for c in clusters):
            raise ValueError("Cannot build a Cluster from Clusters with "
                             "non-homogeneous guards")

        exprs = chain(*[c.exprs for c in clusters])
        ispace = IterationSpace.union(*[c.ispace for c in clusters])

        guards = root.guards

        properties = {}
        for c in clusters:
            for d, v in c.properties.items():
                properties[d] = normalize_properties(properties.get(d, v), v)

        try:
            syncs = normalize_syncs(*[c.syncs for c in clusters])
        except ValueError:
            raise ValueError("Cannot build a Cluster from Clusters with "
                             "non-compatible synchronization operations")

        return Cluster(exprs, ispace, guards, properties, syncs)

    def rebuild(self, *args, **kwargs):
        """
        Build a new Cluster from the attributes given as keywords. All other
        attributes are taken from ``self``.
        """
        # Shortcut for backwards compatibility
        if args:
            if len(args) != 1:
                raise ValueError("rebuild takes at most one positional argument (exprs)")
            if kwargs.get('exprs'):
                raise ValueError("`exprs` provided both as arg and kwarg")
            kwargs['exprs'] = args[0]

        return Cluster(exprs=kwargs.get('exprs', self.exprs),
                       ispace=kwargs.get('ispace', self.ispace),
                       guards=kwargs.get('guards', self.guards),
                       properties=kwargs.get('properties', self.properties),
                       syncs=kwargs.get('syncs', self.syncs))

    @property
    def exprs(self):
        return self._exprs

    @property
    def ispace(self):
        return self._ispace

    @property
    def itintervals(self):
        return self.ispace.itintervals

    @property
    def sub_iterators(self):
        return self.ispace.sub_iterators

    @property
    def directions(self):
        return self.ispace.directions

    @property
    def guards(self):
        return self._guards

    @property
    def properties(self):
        return self._properties

    @property
    def syncs(self):
        return self._syncs

    @cached_property
    def free_symbols(self):
        return set().union(*[e.free_symbols for e in self.exprs])

    @cached_property
    def dimensions(self):
        return set().union(*[i._defines for i in self.ispace.dimensions])

    @cached_property
    def used_dimensions(self):
        """
        The Dimensions that *actually* appear among the expressions in ``self``.
        These do not necessarily coincide the IterationSpace Dimensions; for
        example, reduction or redundant (i.e., invariant) Dimensions won't
        appear in an expression.
        """
        return {i for i in self.free_symbols if i.is_Dimension}

    @cached_property
    def scope(self):
        return Scope(self.exprs)

    @cached_property
    def functions(self):
        return self.scope.functions

    @cached_property
    def has_increments(self):
        return any(e.is_Increment for e in self.exprs)

    @cached_property
    def is_scalar(self):
        return not any(f.is_Function for f in self.scope.writes)

    @cached_property
    def grid(self):
        grids = set(f.grid for f in self.functions if f.is_DiscreteFunction) - {None}
        if len(grids) == 1:
            return grids.pop()
        else:
            raise ValueError("Cluster has no unique Grid")

    @cached_property
    def is_dense(self):
        """
        A Cluster is dense if at least one of the following conditions is True:

            * It is defined over a unique Grid and all of the Grid Dimensions
              are PARALLEL.
            * Only DiscreteFunctions are written and only affine index functions
              are used (e.g., `a[x+1, y-2]` is OK, while `a[b[x], y-2]` is not)
        """
        # Hopefully it's got a unique Grid and all Dimensions are PARALLEL (or
        # at most PARALLEL_IF_PVT). This is a quick and easy check so we try it first
        try:
            pset = {PARALLEL, PARALLEL_IF_PVT}
            grid = self.grid
            for d in grid.dimensions:
                if not any(pset & v for k, v in self.properties.items()
                           if d in k._defines):
                    raise ValueError
            return True
        except ValueError:
            pass

        # Fallback to legacy is_dense checks
        return (not any(e.conditionals for e in self.exprs) and
                not any(f.is_SparseFunction for f in self.functions) and
                all(a.is_regular for a in self.scope.accesses))

    @property
    def is_sparse(self):
        return not self.is_dense

    @cached_property
    def dtype(self):
        """
        The arithmetic data type of the Cluster.

        If the Cluster performs floating point arithmetic, then the expressions
        performing integer arithmetic are ignored, assuming that they are only
        carrying out array index calculations.

        If two expressions perform calculations with different precision, the
        data type with highest precision is returned.
        """
        dtypes = set()
        for i in self.exprs:
            try:
                if np.issubdtype(i.dtype, np.generic):
                    dtypes.add(i.dtype)
            except TypeError:
                # E.g. `i.dtype` is a ctypes pointer, which has no dtype equivalent
                pass

        return infer_dtype(dtypes)

    @cached_property
    def dspace(self):
        """
        Derive the DataSpace of the Cluster from its expressions, IterationSpace,
        and Guards.
        """
        accesses = detect_accesses(self.exprs)

        # Construct the `parts` of the DataSpace, that is a projection of the data
        # space for each Function appearing in `self.exprs`
        parts = {}
        for f, v in accesses.items():
            if f is None:
                continue

            intervals = [Interval(d,
                                  min([sdims_min(i) for i in offs]),
                                  max([sdims_max(i) for i in offs]))
                         for d, offs in v.items()]
            intervals = IntervalGroup(intervals)

            # Factor in the IterationSpace -- if the min/max points aren't zero,
            # then the data intervals need to shrink/expand accordingly
            intervals = intervals.promote(lambda d: d.is_Block)
            shift = self.ispace.intervals.promote(lambda d: d.is_Block)
            intervals = intervals.add(shift)

            # Map SubIterators to the corresponding data space Dimension
            # E.g., `xs -> x -> x0_blk0 -> x` or `t0 -> t -> time`
            intervals = intervals.promote(lambda d: d.is_SubIterator)

            # If the bound of a Dimension is explicitly guarded, then we should
            # shrink the `parts` accordingly
            for d, v in self.guards.items():
                ret = v.find(BaseGuardBoundNext)
                assert len(ret) <= 1
                if len(ret) != 1:
                    continue
                if ret.pop().direction is Forward:
                    intervals = intervals.translate(d, v1=-1)
                else:
                    intervals = intervals.translate(d, 1)
            for d in self.properties:
                if self.properties.is_inbound(d):
                    intervals = intervals.zero(d._defines)

            # Special case: if the factor of a ConditionalDimension has value 1,
            # then we can safely resort to the parent's Interval
            intervals = intervals.promote(lambda d: d.is_Conditional and d.factor == 1)

            parts[f] = intervals

        # Determine the Dimensions requiring shifted min/max points to avoid
        # OOB accesses
        oobs = set()
        for f, v in parts.items():
            for i in v:
                if i.dim.is_Sub:
                    d = i.dim.parent
                else:
                    d = i.dim
                try:
                    if i.lower < 0 or \
                       i.upper > f._size_nodomain[d].left + f._size_halo[d].right:
                        # It'd mean trying to access a point before the
                        # left halo (test0) or after the right halo (test1)
                        oobs.update(d._defines)
                except (KeyError, TypeError):
                    # Unable to detect presence of OOB accesses (e.g., `d` not in
                    # `f._size_halo`, that is typical of indirect accesses `A[B[i]]`)
                    pass

        # Construct the `intervals` of the DataSpace, that is a global,
        # Dimension-centric view of the data space
        intervals = IntervalGroup.generate('union', *parts.values())
        # E.g., `db0 -> time`, but `xi NOT-> x`
        intervals = intervals.promote(lambda d: not d.is_Sub)
        intervals = intervals.zero(set(intervals.dimensions) - oobs)

        return DataSpace(intervals, parts)

    @cached_property
    def ops(self):
        """Number of operations performed at each iteration."""
        return sum(estimate_cost(i) for i in self.exprs)

    @cached_property
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
            if not i.is_AbstractFunction:
                continue
            elif i in self.dspace.parts:
                # Stencils extend the data spaces beyond the iteration spaces
                intervals = self.dspace.parts[i]

                # Assume that invariant dimensions always cause new loads/stores
                invariants = self.ispace.intervals.drop(intervals.dimensions)
                intervals = intervals.generate('union', invariants, intervals)

                # Bundles impact depends on the number of components
                try:
                    v = len(i.components)
                except AttributeError:
                    v = 1

                for n in range(v):
                    ret[(i, mode, n)] = intervals
            else:
                ret[(i, mode, 0)] = self.ispace.intervals
        return ret


class ClusterGroup(tuple):

    """
    An immutable, totally-ordered sequence of Clusters.

    Parameters
    ----------
    clusters : tuple of Clusters
        Input elements.
    ispace : IterationSpace, optional
        The IterationSpace section shared by all `clusters`.
    """

    def __new__(cls, clusters, ispace=None):
        obj = super(ClusterGroup, cls).__new__(cls, flatten(as_tuple(clusters)))
        obj._ispace = ispace
        return obj

    @classmethod
    def concatenate(cls, *cgroups):
        return list(chain(*cgroups))

    @cached_property
    def exprs(self):
        return flatten(c.exprs for c in self)

    @cached_property
    def scope(self):
        return Scope(exprs=self.exprs)

    @cached_property
    def ispace(self):
        return self._ispace

    @cached_property
    def guards(self):
        """The guards of each Cluster in self."""
        return tuple(i.guards for i in self)

    @cached_property
    def syncs(self):
        """The synchronization operations of each Cluster in self."""
        return tuple(i.syncs for i in self)

    @cached_property
    def dspace(self):
        """Return the DataSpace of this ClusterGroup."""
        return DataSpace.union(*[i.dspace.reset() for i in self])

    @cached_property
    def dtype(self):
        """
        The arithmetic data type of this ClusterGroup.

        If at least one Cluster performs floating point arithmetic, then the
        Clusters performing integer arithmetic are ignored.

        If two Clusters perform calculations with different precision, the
        data type with highest precision is returned.
        """
        dtypes = {i.dtype for i in self}

        return infer_dtype(dtypes)

    @cached_property
    def meta(self):
        """
        Returns
        -------
        dtype, DSpace
            The data type and the data space of the ClusterGroup.
        """
        return (self.dtype, self.dspace)

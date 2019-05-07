from collections import OrderedDict

from devito.equation import Eq
from devito.ir import (DataSpace, IterationSpace, Interval, IntervalGroup, Cluster,
                       ClusterGroup, detect_accesses, build_intervals, groupby)
from devito.dse.aliases import collect
from devito.dse.backends import BasicRewriter, dse_pass
from devito.symbolics import (estimate_cost, xreplace_constrained, iq_timeinvariant,
                             xreplace_indices)
from devito.dse.manipulation import (common_subexprs_elimination, collect_nested,
                                     compact_temporaries)
from devito.types import Array, Scalar
from devito.parameters import configuration
from devito.types.dimension import TimeDimension

class AdvancedRewriter(BasicRewriter):

    MIN_COST_ALIAS = 10
    """
    Minimum operation count of a non-scalar alias (i.e., "redundant") expression
    to be lifted into a vector temporary (thus increasing the memory footprint)
    """

    MIN_COST_FACTORIZE = 100
    """
    Minimum operation count of an expression so that aggressive factorization
    is applied.
    """

    def _pipeline(self, state):
        self._extract_time_invariants(state)
        self._eliminate_inter_stencil_redundancies(state)
        self._eliminate_intra_stencil_redundancies(state)
        self._factorize(state)

    @dse_pass
    def _extract_time_invariants(self, cluster, template, with_cse=True, **kwargs):
        """
        Extract time-invariant subexpressions, and assign them to temporaries.
        """
        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
        rule = iq_timeinvariant(cluster.trace)
        costmodel = lambda e: estimate_cost(e) > 0
        processed, found = xreplace_constrained(cluster.exprs, make, rule, costmodel)

        if with_cse:
            leaves = [i for i in processed if i not in found]

            # Search for common sub-expressions amongst them (and only them)
            found = common_subexprs_elimination(found, make)

            # Some temporaries may be droppable at this point
            processed = compact_temporaries(found, leaves)

        return cluster.rebuild(processed)

    @dse_pass
    def _factorize(self, cluster, *args, **kwargs):
        """
        Collect terms in each expr in exprs based on the following heuristic:

            * Collect all literals;
            * Collect all temporaries produced by CSE;
            * If the expression has an operation count higher than
              ``self.MIN_COST_FACTORIZE``, then this is applied recursively until
              no more factorization opportunities are available.
        """

        processed = []
        for expr in cluster.exprs:
            handle = collect_nested(expr)
            cost_handle = estimate_cost(handle)

            if cost_handle >= self.MIN_COST_FACTORIZE:
                handle_prev = handle
                cost_prev = estimate_cost(expr)
                while cost_handle < cost_prev:
                    handle_prev, handle = handle, collect_nested(handle)
                    cost_prev, cost_handle = cost_handle, estimate_cost(handle)
                cost_handle, handle = cost_prev, handle_prev

            processed.append(handle)

        return cluster.rebuild(processed)

    @dse_pass
    def _eliminate_inter_stencil_redundancies(self, cluster, template, **kwargs):
        """
        Search for redundancies across the expressions and expose them
        to the later stages of the optimisation pipeline by introducing
        new temporaries of suitable rank.

        Two type of redundancies are sought:

            * Time-invariants, and
            * Across different space points

        Examples
        ========
        Let ``t`` be the time dimension, ``x, y, z`` the space dimensions. Then:

        1) temp = (a[x,y,z]+b[x,y,z])*c[t,x,y,z]
           >>>
           ti[x,y,z] = a[x,y,z] + b[x,y,z]
           temp = ti[x,y,z]*c[t,x,y,z]

        2) temp1 = 2.0*a[x,y,z]*b[x,y,z]
           temp2 = 3.0*a[x,y,z+1]*b[x,y,z+1]
           >>>
           ti[x,y,z] = a[x,y,z]*b[x,y,z]
           temp1 = 2.0*ti[x,y,z]
           temp2 = 3.0*ti[x,y,z+1]
        """
        if cluster.is_sparse:
            return cluster

        # For more information about "aliases", refer to collect.__doc__
        mapper, aliases = collect(cluster.exprs)

        # Redundancies will be stored in space-varying temporaries
        g = cluster.trace
        indices = g.space_indices
        time_invariants = {v.rhs: g.time_invariant(v) for v in g.values()}

        # Find the candidate expressions
        processed = []
        candidates = OrderedDict()
        for k, v in g.items():
            # Cost check (to keep the memory footprint under control)
            naliases = len(mapper.get(v.rhs, []))
            cost = estimate_cost(v, True)*naliases
            if cost >= self.MIN_COST_ALIAS and (naliases > 1 or time_invariants[v.rhs]):
                candidates[v.rhs] = k
            else:
                processed.append(v)

        # Create alias Clusters and all necessary substitution rules
        # for the new temporaries
        alias_clusters = ClusterGroup()
        rules = OrderedDict()
        for origin, alias in aliases.items():
            if all(i not in candidates for i in alias.aliased):
                continue
            # Construct an iteration space suitable for /alias/
            intervals, sub_iterators, directions = cluster.ispace.args
            intervals = [Interval(i.dim, *alias.relaxed_diameter.get(i.dim, i.limits))
                         for i in cluster.ispace.intervals]
            ispace = IterationSpace(intervals, sub_iterators, directions)

            # Optimization: perhaps we can lift the cluster outside the time dimension
            if all(time_invariants[i] for i in alias.aliased):
                ispace = ispace.project(lambda i: not i.is_Time)

            # Build a symbolic function for /alias/
            intervals = ispace.intervals
            halo = [(abs(intervals[i].lower), abs(intervals[i].upper)) for i in indices]
            function = Array(name=template(), dimensions=indices, halo=halo)
            access = tuple(i - intervals[i].lower for i in indices)
            expression = Eq(function[access], origin)

            # Construct a data space suitable for /alias/
            mapper = detect_accesses(expression)
            parts = {k: IntervalGroup(build_intervals(v)).add(intervals)
                     for k, v in mapper.items() if k}
            dspace = DataSpace([i.zero() for i in intervals], parts)

            # Create a new Cluster for /alias/
            alias_clusters.append(Cluster([expression], ispace, dspace))

            # Add substitution rules
            for aliased, distance in alias.with_distance:
                access = [i - intervals[i].lower + j for i, j in distance if i in indices]
                rules[candidates[aliased]] = function[access]
                rules[aliased] = function[access]

        # Group clusters together if possible
        alias_clusters = groupby(alias_clusters).finalize()
        alias_clusters.sort(key=lambda i: i.is_dense)

        # Switch temporaries in the expression trees
        processed = [e.xreplace(rules) for e in processed]

        return alias_clusters + [cluster.rebuild(processed)]

class SkewingRewriter(BasicRewriter):

    MIN_COST_ALIAS = 10
    """
    Minimum operation count of a non-scalar alias (i.e., "redundant") expression
    to be lifted into a vector temporary (thus increasing the memory footprint)
    """

    MIN_COST_FACTORIZE = 100
    """
    Minimum operation count of an expression so that aggressive factorization
    is applied.
    """

    def _pipeline(self, state):
        self._loop_skew(state)
        self._extract_time_invariants(state)
        self._eliminate_inter_stencil_redundancies(state)
        self._eliminate_intra_stencil_redundancies(state)
        self._factorize(state)

    @dse_pass
    def _loop_skew(self, cluster, template, **kwargs):
        """
        Function to perform only the basic skewing in order to have valid data
        dependences.
        """
        skew_factor = -configuration['skew_factor']
        #skew_factor = 2
        t, mapper = None, {}
        skews = {}

        for dim in cluster.ispace.dimensions:
            if t is not None:
                mapper[dim] = dim + skew_factor*t
                skews[dim] = (skew_factor, t)
            elif dim.is_Time:
                if isinstance(dim, TimeDimension):
                    t = dim
                elif isinstance(dim.parent, TimeDimension):
                    t = dim.parent

        if t is None:
            return cluster

        cluster.skewed_loops = skews
        processed = xreplace_indices(cluster.exprs, mapper)
        return cluster.rebuild(processed)

    @dse_pass
    def _extract_time_invariants(self, cluster, template, with_cse=True, **kwargs):
        """
        Extract time-invariant subexpressions, and assign them to temporaries.
        """
        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
        rule = iq_timeinvariant(cluster.trace)
        costmodel = lambda e: estimate_cost(e) > 0
        processed, found = xreplace_constrained(cluster.exprs, make, rule, costmodel)

        if with_cse:
            leaves = [i for i in processed if i not in found]

            # Search for common sub-expressions amongst them (and only them)
            found = common_subexprs_elimination(found, make)

            # Some temporaries may be droppable at this point
            processed = compact_temporaries(found, leaves)

        return cluster.rebuild(processed)

    @dse_pass
    def _factorize(self, cluster, *args, **kwargs):
        """
        Collect terms in each expr in exprs based on the following heuristic:

            * Collect all literals;
            * Collect all temporaries produced by CSE;
            * If the expression has an operation count higher than
              ``self.MIN_COST_FACTORIZE``, then this is applied recursively until
              no more factorization opportunities are available.
        """

        processed = []
        for expr in cluster.exprs:
            handle = collect_nested(expr)
            cost_handle = estimate_cost(handle)

            if cost_handle >= self.MIN_COST_FACTORIZE:
                handle_prev = handle
                cost_prev = estimate_cost(expr)
                while cost_handle < cost_prev:
                    handle_prev, handle = handle, collect_nested(handle)
                    cost_prev, cost_handle = cost_handle, estimate_cost(handle)
                cost_handle, handle = cost_prev, handle_prev

            processed.append(handle)

        return cluster.rebuild(processed)

    @dse_pass
    def _eliminate_inter_stencil_redundancies(self, cluster, template, **kwargs):
        """
        Search for redundancies across the expressions and expose them
        to the later stages of the optimisation pipeline by introducing
        new temporaries of suitable rank.

        Two type of redundancies are sought:

            * Time-invariants, and
            * Across different space points

        Examples
        ========
        Let ``t`` be the time dimension, ``x, y, z`` the space dimensions. Then:

        1) temp = (a[x,y,z]+b[x,y,z])*c[t,x,y,z]
           >>>
           ti[x,y,z] = a[x,y,z] + b[x,y,z]
           temp = ti[x,y,z]*c[t,x,y,z]

        2) temp1 = 2.0*a[x,y,z]*b[x,y,z]
           temp2 = 3.0*a[x,y,z+1]*b[x,y,z+1]
           >>>
           ti[x,y,z] = a[x,y,z]*b[x,y,z]
           temp1 = 2.0*ti[x,y,z]
           temp2 = 3.0*ti[x,y,z+1]
        """
        if cluster.is_sparse:
            return cluster

        # For more information about "aliases", refer to collect.__doc__
        mapper, aliases = collect(cluster.exprs)

        # Redundancies will be stored in space-varying temporaries
        g = cluster.trace
        indices = g.space_indices
        time_invariants = {v.rhs: g.time_invariant(v) for v in g.values()}

        # Find the candidate expressions
        processed = []
        candidates = OrderedDict()
        for k, v in g.items():
            # Cost check (to keep the memory footprint under control)
            naliases = len(mapper.get(v.rhs, []))
            cost = estimate_cost(v, True)*naliases
            if cost >= self.MIN_COST_ALIAS and (naliases > 1 or time_invariants[v.rhs]):
                candidates[v.rhs] = k
            else:
                processed.append(v)

        # Create alias Clusters and all necessary substitution rules
        # for the new temporaries
        alias_clusters = ClusterGroup()
        rules = OrderedDict()
        for origin, alias in aliases.items():
            if all(i not in candidates for i in alias.aliased):
                continue
            # Construct an iteration space suitable for /alias/
            intervals, sub_iterators, directions = cluster.ispace.args
            intervals = [Interval(i.dim, *alias.relaxed_diameter.get(i.dim, i.limits))
                         for i in cluster.ispace.intervals]
            ispace = IterationSpace(intervals, sub_iterators, directions)

            # Optimization: perhaps we can lift the cluster outside the time dimension
            if all(time_invariants[i] for i in alias.aliased):
                ispace = ispace.project(lambda i: not i.is_Time)

            # Build a symbolic function for /alias/
            intervals = ispace.intervals
            halo = [(abs(intervals[i].lower), abs(intervals[i].upper)) for i in indices]
            function = Array(name=template(), dimensions=indices, halo=halo)
            access = tuple(i - intervals[i].lower for i in indices)
            expression = Eq(function[access], origin)

            # Construct a data space suitable for /alias/
            mapper = detect_accesses(expression)
            parts = {k: IntervalGroup(build_intervals(v)).add(intervals)
                     for k, v in mapper.items() if k}
            dspace = DataSpace([i.zero() for i in intervals], parts)

            # Create a new Cluster for /alias/
            alias_clusters.append(Cluster([expression], ispace, dspace))

            # Add substitution rules
            for aliased, distance in alias.with_distance:
                access = [i - intervals[i].lower + j for i, j in distance if i in indices]
                rules[candidates[aliased]] = function[access]
                rules[aliased] = function[access]

        # Group clusters together if possible
        alias_clusters = groupby(alias_clusters).finalize()
        alias_clusters.sort(key=lambda i: i.is_dense)

        # Switch temporaries in the expression trees
        processed = [e.xreplace(rules) for e in processed]

        return alias_clusters + [cluster.rebuild(processed)]

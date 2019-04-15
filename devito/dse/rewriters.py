import abc
from collections import OrderedDict
from time import time

import numpy as np
from sympy import cos, sin

from devito.equation import Eq
from devito.ir import (DataSpace, IterationSpace, Interval, IntervalGroup, Cluster,
                       ClusterGroup, detect_accesses, build_intervals, groupby)
from devito.dse.aliases import collect
from devito.dse.manipulation import (common_subexprs_elimination, collect_nested,
                                     compact_temporaries)
from devito.logger import dse
from devito.symbolics import (bhaskara_cos, bhaskara_sin, estimate_cost, freeze,
                              iq_timeinvariant, iq_timevarying, pow_to_mul,
                              retrieve_indexed, q_affine, q_leaf, q_scalar,
                              q_sum_of_product, q_terminalop, xreplace_constrained)
from devito.tools import flatten, generator
from devito.types import Array, Scalar

__all__ = ['BasicRewriter', 'AdvancedRewriter', 'AggressiveRewriter', 'CustomRewriter']


class State(object):

    def __init__(self, cluster, template):
        self.clusters = [cluster]
        self.template = template
        # Track performance of each pass
        self.ops = OrderedDict()
        self.timings = OrderedDict()

    def update(self, clusters):
        self.clusters = clusters or self.clusters


def dse_pass(func):

    def wrapper(self, state, **kwargs):
        # Invoke the DSE pass on each Cluster
        tic = time()
        state.update(flatten([func(self, c, state.template, **kwargs)
                              for c in state.clusters]))
        toc = time()

        # Profiling
        key = '%s%d' % (func.__name__, len(state.timings))
        state.timings[key] = toc - tic
        if self.profile:
            candidates = [c.exprs for c in state.clusters if c.is_dense]
            state.ops[key] = estimate_cost(flatten(candidates))

    return wrapper


class AbstractRewriter(object):

    """
    Transform a Cluster of SymPy expressions into one or more clusters with
    reduced operation count.
    """

    __metaclass__ = abc.ABCMeta

    tempname = 'r'
    """
    Prefix of temporary variables.
    """

    def __init__(self, profile=True, template=None):
        self.profile = profile

        # Used to build globally-unique temporaries
        if template is None:
            counter = generator()
            self.template = lambda: "%s%d" % (AbstractRewriter.tempname, counter())
        else:
            assert callable(template)
            self.template = template

        # Track performance of each cluster
        self.run_summary = []

    def run(self, cluster):
        state = State(cluster, self.template)

        self._pipeline(state)

        self._finalize(state)

        if self.profile:
            # Print a summary of the applied transformations
            row = "%s [flops: %s, elapsed: %.2f]"
            summary = " >>\n     ".join(row % ("".join(filter(lambda c: not c.isdigit(),
                                                              k[1:])),
                                               str(state.ops.get(k, "?")), v)
                                        for k, v in state.timings.items())
            elapsed = sum(state.timings.values())
            dse("%s\n     [Total elapsed: %.2f s]" % (summary, elapsed))
            self.run_summary.append({'ops': state.ops, 'timings': state.timings})

        return state.clusters

    @abc.abstractmethod
    def _pipeline(self, state):
        return

    @dse_pass
    def _finalize(self, cluster, *args, **kwargs):
        """
        Finalize the DSE output: ::

            * Pow-->Mul. Convert integer powers in an expression to Muls,
              like a**2 => a*a.
            * Freezing. Make sure that subsequent SymPy operations applied to
              the expressions in ``cluster.exprs`` will not alter the effect of
              the DSE passes.
        """
        exprs = [pow_to_mul(e) for e in cluster.exprs]
        return cluster.rebuild([freeze(e) for e in exprs])


class BasicRewriter(AbstractRewriter):

    def _pipeline(self, state):
        self._eliminate_intra_stencil_redundancies(state)
        self._extract_nonaffine_indices(state)
        self._extract_increments(state)

    @dse_pass
    def _extract_nonaffine_indices(self, cluster, template, **kwargs):
        """
        Extract non-affine array indices, and assign them to temporaries.
        """
        make = lambda: Scalar(name=template(), dtype=np.int32).indexify()

        mapper = OrderedDict()
        for e in cluster.exprs:
            # Note: using mode='all' and then checking for presence in the mapper
            # (a few lines below), rather retrieving unique indexeds only (a set),
            # is the key to deterministic code generation
            for indexed in retrieve_indexed(e, mode='all'):
                for i, d in zip(indexed.indices, indexed.function.indices):
                    if q_affine(i, d) or q_scalar(i):
                        continue
                    elif i not in mapper:
                        mapper[i] = make()

        processed = [Eq(v, k) for k, v in mapper.items()]
        processed.extend([e.xreplace(mapper) for e in cluster.exprs])

        return cluster.rebuild(processed)

    @dse_pass
    def _extract_increments(self, cluster, template, **kwargs):
        """
        Extract the RHS of non-local tensor expressions performing an associative
        and commutative increment, and assign them to temporaries.
        """
        processed = []
        for e in cluster.exprs:
            if e.is_Increment and e.lhs.function.is_Input:
                handle = Scalar(name=template(), dtype=e.dtype).indexify()
                if e.rhs.is_Symbol:
                    extracted = e.rhs
                else:
                    extracted = e.rhs.func(*[i for i in e.rhs.args if i != e.lhs])
                processed.extend([Eq(handle, extracted), e.func(e.lhs, handle)])
            else:
                processed.append(e)

        return cluster.rebuild(processed)

    @dse_pass
    def _eliminate_intra_stencil_redundancies(self, cluster, template, **kwargs):
        """
        Perform common subexpression elimination, bypassing the tensor expressions
        extracted in previous passes.
        """

        skip = [e for e in cluster.exprs if e.lhs.base.function.is_Array]
        candidates = [e for e in cluster.exprs if e not in skip]

        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()

        processed = common_subexprs_elimination(candidates, make)

        return cluster.rebuild(skip + processed)

    @dse_pass
    def _optimize_trigonometry(self, cluster, **kwargs):
        """
        Rebuild ``exprs`` replacing trigonometric functions with Bhaskara
        polynomials.
        """

        processed = []
        for expr in cluster.exprs:
            handle = expr.replace(sin, bhaskara_sin)
            handle = handle.replace(cos, bhaskara_cos)
            processed.append(handle)

        return cluster.rebuild(processed)


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
        Factorize trascendental functions, symbolic powers, numeric coefficients.

        If the expression has an operation count greater than
        ``self.MIN_COST_FACTORIZE``, then the algorithm is applied recursively
        until no more factorization opportunities are detected.
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
        Search aliasing expressions and capture them into vector temporaries.

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
        aliases = collect(cluster.exprs)

        # Redundancies will be stored in space-varying temporaries
        g = cluster.trace
        indices = g.space_indices
        time_invariants = {v.rhs: g.time_invariant(v) for v in g.values()}

        # Find the candidate expressions
        processed = []
        candidates = OrderedDict()
        for k, v in g.items():
            # Cost check (to keep the memory footprint under control)
            naliases = len(aliases.get(v.rhs))
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
                access = [i - intervals[i].lower + distance[i] for i in distance.labels
                          if i in indices]
                rules[candidates[aliased]] = function[access]
                rules[aliased] = function[access]

        # Group clusters together if possible
        alias_clusters = groupby(alias_clusters).finalize()
        alias_clusters.sort(key=lambda i: i.is_dense)

        # Switch temporaries in the expression trees
        processed = [e.xreplace(rules) for e in processed]

        return alias_clusters + [cluster.rebuild(processed)]


class AggressiveRewriter(AdvancedRewriter):

    def _pipeline(self, state):
        # Three CIRE phases, progressively searching for less structure
        self._extract_time_varying(state)
        self._extract_time_invariants(state, with_cse=False)
        self._eliminate_inter_stencil_redundancies(state)

        self._extract_sum_of_products(state)
        self._eliminate_inter_stencil_redundancies(state)
        self._extract_sum_of_products(state)

        self._factorize(state)
        self._eliminate_intra_stencil_redundancies(state)

    @dse_pass
    def _extract_time_varying(self, cluster, template, **kwargs):
        """
        Extract time-varying subexpressions, and assign them to temporaries.
        Time varying subexpressions arise for example when approximating
        derivatives through finite differences.
        """

        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
        rule = iq_timevarying(cluster.trace)
        costmodel = lambda i: estimate_cost(i) > 0
        processed, _ = xreplace_constrained(cluster.exprs, make, rule, costmodel)

        return cluster.rebuild(processed)

    @dse_pass
    def _extract_sum_of_products(self, cluster, template, **kwargs):
        """
        Extract sub-expressions in sum-of-product form, and assign them to temporaries.
        """
        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
        rule = q_sum_of_product
        costmodel = lambda e: not (q_leaf(e) or q_terminalop(e))
        processed, _ = xreplace_constrained(cluster.exprs, make, rule, costmodel)

        return cluster.rebuild(processed)


class CustomRewriter(AggressiveRewriter):

    def _pipeline(self, state):
        raise NotImplementedError

from __future__ import absolute_import

from collections import OrderedDict
from time import time

from sympy import (Eq, Indexed, cos, sin)

from devito.dse.aliases import collect_aliases
from devito.dse.clusterizer import clusterize
from devito.dse.extended_sympy import bhaskara_cos, bhaskara_sin
from devito.dse.inspection import count, estimate_cost, estimate_memory
from devito.dse.manipulation import (collect_nested, freeze_expression,
                                     xreplace_constrained)
from devito.dse.queries import iq_timeinvariant, iq_timevarying, q_op
from devito.interfaces import ScalarFunction, TensorFunction
from devito.logger import dse, dse_warning
from devito.stencil import Stencil
from devito.tools import flatten

__all__ = ['rewrite']

_temp_prefix = 'temp'


def rewrite(clusters, mode='advanced'):
    """
    Transform N :class:`Cluster`s of SymPy expressions into M :class:`Cluster`s
    of SymPy expressions with reduced operation count, with M >= N.

    :param clusters: The clusters to be transformed.
    :param mode: drive the expression transformation as follows: ::

         * 'noop': Do nothing.
         * 'basic': Apply common sub-expressions elimination.
         * 'factorize': Apply heuristic factorization of temporaries.
         * 'approx-trigonometry': Replace expensive trigonometric
             functions with suitable polynomial approximations.
         * 'glicm': Heuristically hoist time-invariant and cross-stencil
             redundancies.
         * 'advanced': Compose all known transformations.
    """
    if not mode:
        return clusters
    elif isinstance(mode, str):
        mode = set([mode])
    else:
        try:
            mode = set(mode)
        except TypeError:
            dse_warning("Arg mode must be str or tuple (got %s)" % type(mode))
            return clusters

    if mode.isdisjoint(set(Rewriter.modes)):
        dse_warning("Unknown rewrite mode(s) %s" % str(mode))
        return clusters
    else:
        processed = []
        for cluster in clusters:
            rewriter = Rewriter(mode, profile=cluster.is_dense)
            processed.extend(rewriter.run(cluster))
        return processed


def dse_pass(func):

    def wrapper(self, state, **kwargs):
        if self.mode.intersection(set(self.triggers[func.__name__])):
            tic = time()
            state.update(flatten([func(self, c) for c in state.clusters]))
            toc = time()

            key = '%s%d' % (func.__name__, len(self.timings))
            self.timings[key] = toc - tic
            if self.profile:
                candidates = [c.exprs for c in state.clusters if c.is_dense]
                self.ops[key] = estimate_cost(flatten(candidates))

    return wrapper


class State(object):

    def __init__(self, cluster):
        self.clusters = [cluster]
        self.mapper = OrderedDict()

    def update(self, clusters):
        self.clusters = clusters or self.clusters

    @property
    def time_invariants(self):
        return [i for i in self.exprs if i.lhs in self.mapper]

    @property
    def time_varying(self):
        return [i for i in self.exprs if i not in self.time_invariants]

    @property
    def ops_time_invariants(self):
        return estimate_cost(self.time_invariants)

    @property
    def ops_time_varying(self):
        return estimate_cost(self.time_varying)

    @property
    def ops(self):
        return self.ops_time_invariants + self.ops_time_varying

    @property
    def memory_time_invariants(self):
        return estimate_memory(self.time_invariants)

    @property
    def memory_time_varying(self):
        return estimate_memory(self.time_varying)

    @property
    def memory(self):
        return self.memory_time_invariants + self.memory_time_varying


class Rewriter(object):

    """
    Transform expressions to reduce their operation count.
    """

    """
    All DSE transformation modes.
    """
    modes = ('noop', 'basic', 'advanced',
             'factorize', 'approx-trigonometry', 'glicm')

    """
    Name conventions for new temporaries.
    """
    conventions = {
        'redundancy': 'r',
        'time-invariant': 'ti',
        'time-dependent': 'td',
        'temporary': 'tcse'
    }

    """
    Track what options trigger a given pass.
    """
    triggers = {
        '_extract_time_varying': ('advanced',),
        '_extract_time_invariants': ('advanced',),
        '_eliminate_intra_stencil_redundancies': ('basic', 'advanced'),
        '_eliminate_inter_stencil_redundancies': ('glicm', 'advanced'),
        '_factorize': ('factorize', 'advanced'),
        '_optimize_trigonometry': ('approx-trigonometry',),
        '_finalize': modes
    }

    """
    Bag of thresholds, to be used to trigger or prevent certain transformations.
    """
    thresholds = {
        'min-cost-space-hoist': 10,
        'min-cost-time-hoist': 200,
        'min-cost-factorize': 100,
        'max-operands': 40,
    }

    def __init__(self, mode, profile=True):
        self.mode = mode
        self.profile = profile

        self.ops = OrderedDict()
        self.timings = OrderedDict()

    def run(self, cluster):
        state = State(cluster)

        self._extract_time_varying(state)
        self._extract_time_invariants(state)
        self._optimize_trigonometry(state)
        self._eliminate_inter_stencil_redundancies(state)
        self._eliminate_intra_stencil_redundancies(state)
        self._factorize(state)

        self._finalize(state)

        self._summary()

        return state.clusters

    @dse_pass
    def _extract_time_varying(self, cluster, **kwargs):
        """
        Extract time-varying subexpressions, and assign them to temporaries.
        Time varying subexpressions arise for example when approximating
        derivatives through finite differences.
        """

        template = self.conventions['time-dependent'] + "%d"
        make = lambda i: ScalarFunction(name=template % i).indexify()

        rule = iq_timevarying(cluster.trace)

        cm = lambda i: estimate_cost(i) > 0

        processed = xreplace_constrained(cluster.exprs, make, rule, cm)

        return cluster.rebuild(processed)

    @dse_pass
    def _extract_time_invariants(self, cluster, **kwargs):
        """
        Extract time-invariant subexpressions, and assign them to temporaries.
        """

        template = self.conventions['time-invariant'] + "%d"
        make = lambda i: ScalarFunction(name=template % i).indexify()

        rule = iq_timeinvariant(cluster.trace)

        cm = lambda e: estimate_cost(e) > 0

        processed = xreplace_constrained(cluster.exprs, make, rule, cm)

        return cluster.rebuild(processed)

    @dse_pass
    def _eliminate_intra_stencil_redundancies(self, cluster, **kwargs):
        """
        Perform common subexpression elimination.
        """

        # Not using SymPy's CSE() function for three reasons:
        # - capture index functions (we are not interested in integer arithmetic)
        # - doesn't consider the possibliity of losing factorization opportunities
        # - very slow

        skip = [e for e in cluster.exprs if e.lhs.base.function.is_SymbolicFunction]
        candidates = [e for e in cluster.exprs if e not in skip]

        template = self.conventions['temporary'] + "%d"

        mapped = []
        while True:
            # Detect redundancies
            counted = count(mapped + candidates, q_op).items()
            targets = OrderedDict([(k, estimate_cost(k)) for k, v in counted if v > 1])
            if not targets:
                break

            # Create temporaries
            make = lambda i: ScalarFunction(name=template % (len(mapped) + i)).indexify()
            highests = [k for k, v in targets.items() if v == max(targets.values())]
            mapper = OrderedDict([(e, make(i)) for i, e in enumerate(highests)])
            candidates = [e.xreplace(mapper) for e in candidates]
            mapped = [e.xreplace(mapper) for e in mapped]
            mapped = [Eq(v, k) for k, v in reversed(mapper.items())] + mapped

            # Prepare for the next round
            for k in highests:
                targets.pop(k)
        processed = mapped + candidates

        # Simply renumber the temporaries in ascending order
        mapper = {i.lhs: j.lhs for i, j in zip(mapped, reversed(mapped))}
        processed = [e.xreplace(mapper) for e in processed]

        return cluster.rebuild(skip + processed)

    @dse_pass
    def _factorize(self, cluster, **kwargs):
        """
        Collect terms in each expr in exprs based on the following heuristic:

            * Collect all literals;
            * Collect all temporaries produced by CSE;
            * If the expression has an operation count higher than
              self.threshold, then this is applied recursively until
              no more factorization opportunities are available.
        """

        processed = []
        for expr in cluster.exprs:
            handle = collect_nested(expr)
            cost_handle = estimate_cost(handle)

            if cost_handle >= self.thresholds['min-cost-factorize']:
                handle_prev = handle
                cost_prev = estimate_cost(expr)
                while cost_handle < cost_prev:
                    handle_prev, handle = handle, collect_nested(handle)
                    cost_prev, cost_handle = cost_handle, estimate_cost(handle)
                cost_handle, handle = cost_prev, handle_prev

            processed.append(handle)

        return cluster.rebuild(processed)

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

    @dse_pass
    def _eliminate_inter_stencil_redundancies(self, cluster, **kwargs):
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

        # For more information about "aliases", refer to collect_aliases.__doc__
        mapper, aliases = collect_aliases(cluster.exprs)

        # Redundancies will be stored in space-varying temporaries
        g = cluster.trace
        indices = g.space_indices
        shape = g.space_shape

        # Template for captured redundancies
        name = self.conventions['redundancy'] + "%d"
        template = lambda i: TensorFunction(name=name % i, shape=shape,
                                            dimensions=indices).indexed

        # Cross-time redundancies, if any, cannot be exploited by this pass
        scope = cluster.stencil.section(indices)
        if len(scope) == 0:
            # No time dimension, nothing to capture
            return cluster
        elif len(scope) > 1:
            dse_warning("Unexpected Stencil %s" % str(cluster.stencil))
            return cluster
        else:
            dim = scope.dimensions[0]
            for origin, alias in list(aliases.items()):
                if alias.stencil.get(dim) != {0}:
                    # Time-varying, let's not exploit this
                    aliases.pop(origin)
                    for i in alias.aliased:
                        mapper.pop(i)
                elif dim in alias.stencil:
                    aliases[origin] = alias.relax(Stencil.union(*[scope, alias.stencil]))

        # Find the candidate expressions
        processed = []
        candidates = OrderedDict()
        for k, v in g.items():
            # Cost check (to keep the memory footprint under control)
            naliases = len(mapper.get(v.rhs, []))
            cost = estimate_cost(v, True)*naliases
            if cost >= self.thresholds['min-cost-time-hoist'] and g.time_invariant(v):
                candidates[v.rhs] = k
            elif cost >= self.thresholds['min-cost-space-hoist'] and naliases > 1:
                candidates[v.rhs] = k
            else:
                processed.append(Eq(k, v.rhs))

        # TODO need a way of incorporating the time dimension in the stencil

        # Create temporaries capturing redundant computation
        found = []
        rules = OrderedDict()
        stencils = OrderedDict()
        for c, (origin, alias) in enumerate(aliases.items()):
            temporary = Indexed(template(c), *indices)
            found.append(Eq(temporary, origin))
            # Track the stencil of each TensorFunction introduced
            stencils[temporary] = alias.stencil
            for aliased, distance in alias.with_distance:
                coordinates = [sum([i, j]) for i, j in distance.items() if i in indices]
                rules[candidates[aliased]] = Indexed(template(c), *tuple(coordinates))

        # Create the alias clusters
        alias_clusters = clusterize(found, stencils)
        alias_clusters = sorted(alias_clusters, key=lambda i: i.is_dense)

        # Switch temporaries in the expression trees
        processed = [e.xreplace(rules) for e in processed]

        return alias_clusters + [cluster.rebuild(processed)]

    @dse_pass
    def _finalize(self, cluster, **kwargs):
        """
        Finalize the DSE output: ::

            * Freezing. Make sure that subsequent SymPy operations applied to
              the expressions in ``cluster.exprs`` will not alter the effect of
              the DSE passes.
        """
        return cluster.rebuild([freeze_expression(e) for e in cluster.exprs])

    def _summary(self):
        """
        Print a summary of the DSE transformations
        """

        if self.profile:
            row = "%s [flops: %s, elapsed: %.2f]"
            summary = " >>\n     ".join(row % (filter(lambda c: not c.isdigit(), k[1:]),
                                               str(self.ops.get(k, "?")), v)
                                        for k, v in self.timings.items())
            elapsed = sum(self.timings.values())
            dse("%s\n     [Total elapsed: %.2f s]" % (summary, elapsed))

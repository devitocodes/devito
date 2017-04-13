"""
The Devito symbolic engine is built on top of SymPy and provides two
classes of functions:
- for inspection of expressions
- for (in-place) manipulation of expressions
- for creation of new objects given some expressions
All exposed functions are prefixed with 'dse' (devito symbolic engine)
"""

from __future__ import absolute_import

from collections import OrderedDict, Sequence
from time import time

from sympy import (Eq, Indexed, cos, sin)

from devito.dse.aliases import collect_aliases
from devito.dse.extended_sympy import bhaskara_cos, bhaskara_sin
from devito.dse.graph import Cluster, temporaries_graph
from devito.dse.inspection import count, estimate_cost, estimate_memory
from devito.dse.manipulation import (collect_nested, freeze_expression,
                                     xreplace_constrained)
from devito.dse.queries import iq_timeinvariant, iq_timevarying, q_op, q_indexed
from devito.interfaces import ScalarFunction, TensorFunction
from devito.logger import dse, dse_warning
from devito.stencil import Stencil
from devito.tools import flatten

__all__ = ['rewrite']

_temp_prefix = 'temp'


def rewrite(expr, mode='advanced'):
    """
    Transform expressions to reduce their operation count.

    :param expr: the target expression.
    :param mode: drive the expression transformation as follows: ::

         * 'noop': do nothing, but track performance metrics
         * 'basic': apply common sub-expressions elimination.
         * 'factorize': apply heuristic factorization of temporaries.
         * 'approx-trigonometry': replace expensive trigonometric
             functions with suitable polynomial approximations.
         * 'glicm': apply heuristic hoisting of time-invariant terms.
         * 'advanced': compose all known transformations.
    """

    if isinstance(expr, Sequence):
        assert all(isinstance(e, Eq) for e in expr)
        expr = list(expr)
    elif isinstance(expr, Eq):
        expr = [expr]
    else:
        raise ValueError("Got illegal expr of type %s." % type(expr))

    if not mode:
        return State(expr)
    elif isinstance(mode, str):
        mode = set([mode])
    else:
        try:
            mode = set(mode)
        except TypeError:
            dse_warning("Arg mode must be str or tuple (got %s)" % type(mode))
            return State(expr)
    if mode.isdisjoint(set(Rewriter.modes)):
        dse_warning("Unknown rewrite mode(s) %s" % str(mode))
        return State(expr)
    else:
        return Rewriter(expr).run(mode)


def dse_pass(func):

    def wrapper(self, state, **kwargs):
        if kwargs['mode'].intersection(set(self.triggers[func.__name__])):
            tic = time()
            state.update(**func(self, state))
            toc = time()

            key = '%s%d' % (func.__name__, len(self.timings))
            self.timings[key] = toc - tic
            if self.profile:
                # Only count operations of those expressions that will be executed
                # at every space-time iteration
                traces = [c.trace for c in state.clusters]
                exprs = flatten(i.values() for i in traces
                                if i.space_indices and not i.time_invariant())
                self.ops[key] = estimate_cost(exprs)

    return wrapper


class State(object):

    def __init__(self, exprs):
        self.input = exprs
        self.exprs = exprs
        self.mapper = OrderedDict()

        # Compute the stencil of each tensor expression
        mapper = OrderedDict()
        for expr in exprs:
            mapper.setdefault(expr.lhs, []).append(expr)
        domains = OrderedDict()
        for k, v in mapper.items():
            domains[k] = Stencil.union(*[Stencil(i) for i in v])
        self.domains = domains

    def update(self, exprs=None, domains=None):
        self.exprs = exprs or self.exprs
        self.domains.update(domains or {})

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

    @property
    def output_fields(self):
        return [i.lhs.base.function for i in self.input if q_indexed(i.lhs)]

    @property
    def clusters(self):
        """
        Clusterize the expressions in ``self.exprs``. For more information
        about clusters, refer to TemporariesGraph.clusters.
        """
        clusters = temporaries_graph(self.exprs).clusters(self.domains)
        return Cluster.merge(clusters)


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

    def __init__(self, exprs, profile=True):
        self.exprs = exprs

        self.profile = profile
        self.ops = OrderedDict()
        self.timings = OrderedDict()

    def run(self, mode):
        state = State(self.exprs)

        self._extract_time_varying(state, mode=mode)
        self._extract_time_invariants(state, mode=mode)
        self._optimize_trigonometry(state, mode=mode)
        self._eliminate_inter_stencil_redundancies(state, mode=mode)
        self._eliminate_intra_stencil_redundancies(state, mode=mode)
        self._factorize(state, mode=mode)

        self._finalize(state, mode=mode)

        self._summary(mode)

        return state

    @dse_pass
    def _extract_time_varying(self, state, **kwargs):
        """
        Extract time-varying subexpressions, and assign them to temporaries.
        Time varying subexpressions arise for example when approximating
        derivatives through finite differences.
        """

        template = self.conventions['time-dependent'] + "%d"
        make = lambda i: ScalarFunction(name=template % i).indexify()

        rule = iq_timevarying(state.exprs)

        cm = lambda i: estimate_cost(i) > 0

        processed = xreplace_constrained(state.exprs, make, rule, cm)

        return {'exprs': processed}

    @dse_pass
    def _extract_time_invariants(self, state, **kwargs):
        """
        Extract time-invariant subexpressions, and assign them to temporaries.
        """

        template = self.conventions['time-invariant'] + "%d"
        make = lambda i: ScalarFunction(name=template % i).indexify()

        rule = iq_timeinvariant(state.exprs)

        cm = lambda e: estimate_cost(e) > 0

        processed = xreplace_constrained(state.exprs, make, rule, cm)

        return {'exprs': processed}

    @dse_pass
    def _eliminate_intra_stencil_redundancies(self, state, **kwargs):
        """
        Perform common subexpression elimination.
        """

        # Not using SymPy's CSE() function for three reasons:
        # - capture index functions (we are not interested in integer arithmetic)
        # - doesn't consider the possibliity of losing factorization opportunities
        # - very slow

        skip = [e for e in state.exprs if e.lhs.base.function.is_TensorFunction]
        candidates = [e for e in state.exprs if e not in skip]

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

        return {'exprs': skip + processed}

    @dse_pass
    def _factorize(self, state, **kwargs):
        """
        Collect terms in each expr in exprs based on the following heuristic:

            * Collect all literals;
            * Collect all temporaries produced by CSE;
            * If the expression has an operation count higher than
              self.threshold, then this is applied recursively until
              no more factorization opportunities are available.
        """

        processed = []
        for expr in state.exprs:
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

        return {'exprs': processed}

    @dse_pass
    def _optimize_trigonometry(self, state, **kwargs):
        """
        Rebuild ``exprs`` replacing trigonometric functions with Bhaskara
        polynomials.
        """

        processed = []
        for expr in state.exprs:
            handle = expr.replace(sin, bhaskara_sin)
            handle = handle.replace(cos, bhaskara_cos)
            processed.append(handle)

        return {'exprs': processed}

    @dse_pass
    def _eliminate_inter_stencil_redundancies(self, state, **kwargs):
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

        graph = temporaries_graph(state.exprs)
        indices = graph.space_indices
        shape = graph.space_shape

        # For more information about "aliases", refer to collect_aliases.__doc__
        mapper, aliases = collect_aliases(state.exprs)

        # Template for captured redundancies
        name = self.conventions['redundancy'] + "%d"
        template = lambda i: TensorFunction(name=name % i, shape=shape,
                                            dimensions=indices).indexed

        # Retain only the expensive time-invariant expressions (to minimize memory)
        processed = []
        candidates = OrderedDict()
        for k, v in graph.items():
            naliases = len(mapper.get(v.rhs, [v]))
            cost = estimate_cost(v, True)*naliases
            if cost >= self.thresholds['min-cost-time-hoist']\
                    and graph.time_invariant(v):
                candidates[v.rhs] = k
            elif cost >= self.thresholds['min-cost-space-hoist'] and naliases > 1:
                candidates[v.rhs] = k
            else:
                processed.append(Eq(k, v.rhs))

        # Create temporaries capturing redundant computation
        found = []
        rules = OrderedDict()
        domains = OrderedDict()
        for c, (origin, alias) in enumerate(aliases.items()):
            temporary = Indexed(template(c), *indices)
            found.append(Eq(temporary, origin))
            # Track the domain of each TensorFunction introduced
            domains[temporary] = alias.stencil
            for aliased, distance in alias.with_distance:
                coordinates = [sum([i, j]) for i, j in distance.items() if i in indices]
                rules[candidates[aliased]] = Indexed(template(c), *tuple(coordinates))

        # Switch temporaries in the expression trees
        processed = found + [e.xreplace(rules) for e in processed]

        return {'exprs': processed, 'domains': domains}

    @dse_pass
    def _finalize(self, state, **kwargs):
        """
        Finalize the DSE output: ::

            * Reordering. The expressions in ``state.exprs`` are reordered so
              that all tensor temporaries will appear at the top, which honors
              the computation semantics.
            * Freezing. Make sure that subsequent SymPy operations applied to
              the expressions in ``state.exprs`` will not alter the effect of
              the DSE passes.
        """
        # Reordering
        graph = temporaries_graph(state.exprs)

        def key(i):
            return (not i.lhs.base.function.is_TensorFunction,
                    not graph.time_invariant(i.rhs),
                    state.exprs.index(i))

        processed = sorted(state.exprs, key=key)

        # Freezing
        processed = [freeze_expression(e) for e in processed]

        return {'exprs': processed}

    def _summary(self, mode):
        """
        Print a summary of the DSE transformations
        """

        if mode.intersection({'basic', 'advanced'}):
            row = "%s [flops: %s, elapsed: %.2f]"
            summary = " >>\n     ".join(row % (filter(lambda c: not c.isdigit(), k[1:]),
                                               str(self.ops.get(k, "?")), v)
                                        for k, v in self.timings.items())
            elapsed = sum(self.timings.values())
            dse("%s\n     [Total elapsed: %.2f s]" % (summary, elapsed))

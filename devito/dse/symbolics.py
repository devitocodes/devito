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

from sympy import (Eq, Indexed, IndexedBase, Symbol, cos,
                   numbered_symbols, preorder_traversal, sin)

from devito.dimension import t, x, y, z
from devito.dse.extended_sympy import bhaskara_cos, bhaskara_sin
from devito.dse.graph import Cluster, Temporary, temporaries_graph
from devito.dse.inspection import (as_symbol, collect_aliases, estimate_cost,
                                   estimate_memory, is_terminal_op, terminals)
from devito.dse.manipulation import (collect_nested, freeze_expression,
                                     filter_expressions, xreplace_constrained)
from devito.interfaces import ScalarFunction, TensorFunction
from devito.logger import dse, dse_warning
from devito.tools import flatten

__all__ = ['rewrite']

_temp_prefix = 'temp'


def rewrite(expr, mode='advanced'):
    """
    Transform expressions to reduce their operation count.

    :param expr: the target expression.
    :param mode: drive the expression transformation. Available modes are
                 'basic', 'factorize', 'approx-trigonometry' and 'advanced'
                 (default). They act as follows: ::

                     * 'noop': do nothing, but track performance metrics
                     * 'basic': apply common sub-expressions elimination.
                     * 'factorize': apply heuristic factorization of temporaries.
                     * 'approx-trigonometry': replace expensive trigonometric
                         functions with suitable polynomial approximations.
                     * 'glicm': apply heuristic hoisting of time-invariant terms.
                     * 'split': split long expressions into smaller sub-expressions
                          exploiting associativity and commutativity.
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
    if mode.isdisjoint({'noop', 'basic', 'factorize', 'approx-trigonometry',
                        'glicm', 'advanced'}):
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
        self.aliases = OrderedDict()

    def update(self, exprs=None, aliases=None):
        self.exprs = exprs or self.exprs
        self.aliases = aliases or self.aliases

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
        return [i.lhs for i in self.exprs if isinstance(i.lhs, Indexed)]

    @property
    def exprs_aliased(self):
        return [e for e in self.exprs if e.rhs in self.aliases]

    @property
    def clusters(self):
        """
        Clusterize the expressions in ``self.exprs``. For more information
        about clusters, refer to TemporariesGraph.clusters.
        """
        clusters = temporaries_graph(self.exprs).clusters(self.aliases)
        return Cluster.merge(clusters, self.aliases)


class Rewriter(object):

    """
    Transform expressions to reduce their operation count.
    """

    """
    Name conventions for new temporaries
    """
    conventions = {
        'redundancy': 'r',
        'time-invariant': 'ti',
        'time-dependent': 'td',
        'temporary': 't'
    }

    """
    Track what options trigger a given transformation.
    """
    triggers = {
        '_extract_time_varying': ('advanced',),
        '_extract_time_invariants': ('advanced',),
        '_eliminate_intra_stencil_redundancies': ('basic', 'advanced'),
        '_eliminate_inter_stencil_redundancies': ('glicm', 'advanced'),
        '_factorize': ('factorize', 'advanced'),
        '_optimize_trigonometry': ('approx-trigonometry', 'advanced')
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

        self._finalize(state)

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

        graph = temporaries_graph(state.exprs)
        rule = lambda i: i.is_Number or not graph.time_invariant(i)

        cm = lambda i: estimate_cost(i) > 0

        processed = xreplace_constrained(state.exprs, make, rule, cm)

        processed = filter_expressions(processed, make)

        return {'exprs': processed}

    @dse_pass
    def _extract_time_invariants(self, state, **kwargs):
        """
        Extract time-invariant subexpressions, and assign them to temporaries.
        """

        template = self.conventions['time-invariant'] + "%d"
        make = lambda i: ScalarFunction(name=template % i).indexify()

        graph = temporaries_graph(state.exprs)
        rule = lambda i: not i.is_Number and graph.time_invariant(i)

        cm = lambda e: estimate_cost(e) > 0

        processed = xreplace_constrained(state.exprs, make, rule, cm, repeat=True)

        processed = filter_expressions(processed, make)

        return {'exprs': processed}

    @dse_pass
    def _eliminate_intra_stencil_redundancies(self, state, **kwargs):
        """
        Perform common subexpression elimination.
        """

        # Not using SymPy's CSE() function for two reasons:
        # - capture index functions (we are not interested in integer arithmetic)
        # - very slow

        aliased = state.exprs_aliased
        candidates = [e for e in state.exprs if e not in aliased]

        template = self.conventions['temporary'] + "%d"

        def rule(e):
            try:
                as_symbol(e)
                return True
            except TypeError:
                return is_terminal_op(e) or e.is_Function

        cm = lambda e: estimate_cost(e) > 0

        redundancies = []
        while True:
            make = lambda i: \
                ScalarFunction(name=template % (len(redundancies) + i)).indexify()
            handle = xreplace_constrained(candidates, make, rule, cm)

            # Find redundant leaf operations
            found = filter_expressions(handle, make, count=2)

            # Replace redundancies
            mapper = {e.rhs: e.lhs for e in found}
            replaced = [e.xreplace(mapper) for e in candidates]
            if replaced == candidates:
                break

            redundancies += found
            candidates = replaced

        make = lambda i: ScalarFunction(name=template % i).indexify()
        processed = filter_expressions(aliased + redundancies + candidates, make)

        return {'exprs': processed}

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
        space_indices = graph.space_indices

        # For more information about "aliases", refer to collect_aliases.__doc__
        mapper, aliases = collect_aliases([e.rhs for e in state.exprs])

        # Template for captured redundancies
        name = self.conventions['redundancy'] + "%d"
        template = lambda i: TensorFunction(name=name % i, shape=graph.space_shape,
                                            dimensions=space_indices).indexed

        # Retain only the expensive time-invariant expressions (to minimize memory)
        processed = []
        candidates = OrderedDict()
        for k, v in graph.items():
            naliases = len(mapper.get(v.rhs, [v]))
            cost = estimate_cost(v, True)*naliases
            if graph.is_index(k):
                processed.append(Eq(k, v.rhs))
            elif cost >= self.thresholds['min-cost-time-hoist']\
                    and graph.time_invariant(v):
                candidates[v.rhs] = k
            elif cost >= self.thresholds['min-cost-space-hoist'] and naliases > 1:
                candidates[v.rhs] = k
            else:
                processed.append(Eq(k, v.rhs))

        # Create temporaries capturing redundant computation
        c = len(state.aliases)
        found = []
        rules = OrderedDict()
        for origin, info in aliases.items():
            handle = [(v, k) for k, v in candidates.items() if k in info.aliased]
            if handle:
                eq = Eq(Indexed(template(c), *space_indices), origin)
                found.append(freeze_expression(eq))
                for k, v in handle:
                    translation = mapper[v][v]
                    coordinates = tuple(sum([i, j]) for i, j in translation
                                        if i in space_indices)
                    rules[k] = Indexed(template(c), *coordinates)
                c += 1

        # Switch temporaries in the expression trees
        processed = found + [e.xreplace(rules) for e in processed]

        # Only track what is strictly necessary for later passes
        aliases = OrderedDict([(freeze_expression(k), v.offsets)
                               for k, v in aliases.items() if k in candidates])
        aliases = OrderedDict(state.aliases.items() + aliases.items())

        return {'exprs': processed, 'aliases': aliases}

    def _finalize(self, state):
        """
        Reorder the expressions to match the semantics of the provided input, as
        multiple DSE passes might have introduced/altered/removed expressions.

        Also make sure that subsequent sympy operations applied to the expressions
        will not alter the effect of the DSE passes.
        """
        exprs = [freeze_expression(e) for e in state.exprs]

        expected = ['alias-time-invariant', 'alias-time-dependent', 'other']
        graph = temporaries_graph(exprs)
        def key(i):
            if i.rhs in state.aliases:
                index = state.aliases.keys().index(i.rhs)
                if graph.time_invariant(i.rhs):
                    return (expected.index('alias-time-invariant'), index)
                else:
                    return (expected.index('alias-time-dependent'), index)
            else:
                return (expected.index('other'), 0)
        processed = sorted(exprs, key=key)

        state.update(exprs=processed)

    def _summary(self, mode):
        """
        Print a summary of the DSE transformations
        """

        if mode.intersection({'basic', 'advanced'}):
            row = "%s [flops: %d, elapsed: %.2f]"
            summary = " >>\n     ".join(row % (filter(lambda c: not c.isdigit(), k[1:]),
                                               self.ops.get(k, ""), v)
                                        for k, v in self.timings.items())
            elapsed = sum(self.timings.values())
            dse("%s\n     [Total elapsed: %.2f s]" % (summary, elapsed))

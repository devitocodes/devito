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
from sympy.simplify.cse_main import tree_cse

from devito.dimension import t, x, y, z
from devito.dse.extended_sympy import bhaskara_cos, bhaskara_sin
from devito.dse.graph import Temporary, temporaries_graph
from devito.dse.inspection import (collect_aliases, estimate_cost, estimate_memory,
                                   is_binary_op, is_terminal_op, terminals)
from devito.dse.manipulation import (collect_nested, freeze_expression,
                                     xreplace_constrained, xreplace_recursive)
from devito.interfaces import ScalarFunction, TensorFunction
from devito.logger import dse, dse_warning

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
                self.ops[key] = estimate_cost(state.exprs)

    return wrapper


class State(object):

    def __init__(self, exprs):
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
    def clusters(self):
        """
        Clusterize the expressions in ``self.exprs``. For more information
        about clusters, refer to TemporariesGraph.clusters.
        """
        return temporaries_graph(self.exprs).clusters


class Rewriter(object):

    """
    Transform expressions to reduce their operation count.
    """

    """
    Name conventions for new temporaries
    """
    conventions = {
        'cse': 't',
        'redundancy': 'r',
        'time-invariant': 'ti',
        'time-dependent': 'td',
        'split': 'ts'
    }

    """
    Track what options trigger a given transformation.
    """
    triggers = {
        '_cse': ('basic', 'advanced'),
        '_factorize': ('factorize', 'advanced'),
        '_optimize_trigonometry': ('approx-trigonometry',),
        '_capture_redundancies': ('glicm', 'advanced'),
        '_split_expressions': ('split', 'devito3.0')  # TODO: -> 'advanced' upon release
    }

    """
    Bag of thresholds, to be used to trigger or prevent certain transformations.
    """
    thresholds = {
        'min-cost-space-hoist': 10,
        'min-cost-time-hoist': 100,
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

        self._cse(state, mode=mode)
        self._optimize_trigonometry(state, mode=mode)
        self._split_expressions(state, mode=mode)
        self._factorize(state, mode=mode)
        for _ in range(2):
            self._capture_redundancies(state, mode=mode)
        from IPython import embed; embed()

        self._finalize(state)

        self._summary(mode)

        return state

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
    def _cse(self, state, **kwargs):
        """
        Perform common subexpression elimination.
        """

        template = self.conventions['cse']
        temporaries, leaves = tree_cse(state.exprs, numbered_symbols(template))
        for i in range(len(state.exprs)):
            leaves[i] = Eq(state.exprs[i].lhs, leaves[i].rhs)

        # Restore some of the common sub-expressions that have potentially
        # been collected: simple index calculations (eg, t - 1), IndexedBase,
        # Indexed, binary Add, binary Mul.
        revert = OrderedDict()
        keep = OrderedDict()
        for k, v in temporaries:
            if isinstance(v, (IndexedBase, Indexed)):
                revert[k] = v
            elif v.is_Add and not set([t, x, y, z]).isdisjoint(set(v.args)):
                revert[k] = v
            elif is_binary_op(v):
                revert[k] = v
            else:
                keep[k] = v
        for k, v in revert.items():
            mapper = {}
            for i in preorder_traversal(v):
                if isinstance(i, Indexed):
                    new_indices = []
                    for index in i.indices:
                        if index in revert:
                            new_indices.append(revert[index])
                        else:
                            new_indices.append(index)
                    if i.base.label in revert:
                        mapper[i] = Indexed(revert[i.base.label], *new_indices)
                if i in revert:
                    mapper[i] = revert[i]
            revert[k] = v.xreplace(mapper)
        mapper = {}
        for e in leaves + list(keep.values()):
            for i in preorder_traversal(e):
                if isinstance(i, Indexed):
                    new_indices = []
                    for index in i.indices:
                        if index in revert:
                            new_indices.append(revert[index])
                        else:
                            new_indices.append(index)
                    if i.base.label in revert:
                        mapper[i] = Indexed(revert[i.base.label], *new_indices)
                    elif tuple(new_indices) != i.indices:
                        mapper[i] = Indexed(i.base, *new_indices)
                if i in revert:
                    mapper[i] = revert[i]
        leaves = xreplace_recursive(leaves, mapper)
        kept = xreplace_recursive([Eq(k, v) for k, v in keep.items()], mapper)

        # If the RHS of a temporary variable is the LHS of a leaf,
        # update the value of the temporary variable after the leaf
        new_leaves = []
        for leaf in leaves:
            new_leaves.append(leaf)
            for i in kept:
                if leaf.lhs in preorder_traversal(i.rhs):
                    new_leaves.append(i)
                    break

        # Reshuffle to make sure temporaries come later than their read values
        processed = OrderedDict([(i.lhs, i) for i in kept + new_leaves])
        temporaries = set(processed.keys())
        ordered = OrderedDict()
        while processed:
            k, v = processed.popitem(last=False)
            temporary_reads = terminals(v.rhs) & temporaries - {v.lhs}
            if all(i in ordered for i in temporary_reads):
                ordered[k] = v
            else:
                # Must wait for some earlier temporaries, push back into queue
                processed[k] = v

        return {'exprs': list(ordered.values())}

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
    def _capture_redundancies(self, state, **kwargs):
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
                processed.append(v)
            elif cost >= self.thresholds['min-cost-time-hoist']\
                    and graph.time_invariant(v):
                candidates[v.rhs] = k
            elif cost >= self.thresholds['min-cost-space-hoist'] and naliases > 1:
                candidates[v.rhs] = k
            else:
                processed.append(v)

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

    @dse_pass
    def _split_expressions(self, state, **kwargs):
        """
        Split expressions as sum of "canonical" products. Each canonical product
        is assigned to a different temporary. A canonical product has the form: ::

            temp = (sum_i w_i (sum_j u_i[f(t, x, y, z, j)])) g(x, y, z)

        Where ``w_i`` is a real number, ``u_i`` is an indexed time-varying object
        (note that ``t`` appears amongst its indices), ``g`` is a time-independent
        object (only the space dimensions ``x, y, z`` may appear in an indexed
        object within ``g``).

        The output is the input expression itself if found to be non-reducible
        to a sum of canonical products (e.g., if ``u_i`` appeared in denominators).
        """

        # Split the time-dependent sub-expressions
        graph = temporaries_graph(state.exprs)
        rule = lambda i: i.is_Number or not graph.time_invariant(i)
        c = 0

        def cm(e):
            if e.is_Mul and is_terminal_op(e):
                # Do not split individual products
                return False
            return estimate_cost(e) > 0

        processed = []
        for expr in state.exprs:
            processing = expr
            while True:
                if not (processing.rhs.is_Add or processing.rhs.is_Mul):
                    break
                template = self.conventions['time-dependent'] + "%d"
                make = lambda i: ScalarFunction(name=template % (c + len(i))).indexify()
                handle, flag, mapped = xreplace_constrained(processing, make, rule, cm)
                if flag or not mapped:
                    break
                else:
                    processing = handle
                    for k, v in mapped.items():
                        processed.append(Eq(k, v))
                        graph[k] = Temporary(k, v)
                        c += 1
            processed.append(processing)

        # Split the time-invariant sub-expressions
        rule = lambda i: not i.is_Number and graph.time_invariant(i)
        cm = lambda e: estimate_cost(e) > 0
        c = 0
        processed, exprs = [], list(processed)
        for expr in exprs:
            template = self.conventions['time-invariant'] + "%d"
            make = lambda i: ScalarFunction(name=template % (c + len(i))).indexify()
            handle, flag, mapped = xreplace_constrained(expr, make, rule, cm)
            if flag:
                processed.append(expr)
            else:
                processed.extend([Eq(k, v) for k, v in mapped.items()] + [handle])
                c += len(mapped)

        # Split excessively long expressions (improves compilation speed)
        c = 0
        processed, exprs = [], list(processed)
        for expr in exprs:
            if len(terminals(expr)) < self.thresholds['max-operands'] or\
                    not (expr.rhs.is_Add or expr.rhs.is_Mul):
                processed.append(expr)
                continue
            chunks = expr.rhs.args
            template = self.conventions['split'] + "%d"
            targets = [Symbol(template % (c + i)) for i in range(len(chunks))]
            chunks = [Eq(k, v) for k, v in zip(targets, chunks)]
            chunks.append(Eq(expr.lhs, expr.rhs.func(*targets)))
            processed.extend(chunks)
            c += len(chunks)

        return {'exprs': processed}

    def _finalize(self, state):
        """
        Make sure that any subsequent sympy operation applied to the expressions
        in ``state.exprs`` does not alter the structure of the transformed objects.
        """
        state.update(exprs=[freeze_expression(e) for e in state.exprs])

    def _summary(self, mode):
        """
        Print a summary of the DSE transformations
        """

        if mode.intersection({'basic', 'advanced'}):
            summary = " --> ".join("(%s) %s" % (filter(lambda c: not c.isdigit(), k),
                                                str(self.ops.get(k, "")))
                                   for k, v in self.timings.items())
            try:
                # The state after CSE should be used as baseline for fairness
                baseline = self.ops['_cse0']
                gain = float(baseline) / list(self.ops.values())[-1]
                summary = " %s flops; gain: %.2f X" % (summary, gain)
            except (KeyError, ZeroDivisionError):
                pass
            elapsed = sum(self.timings.values())
            dse("%s [%.2f s]" % (summary, elapsed))

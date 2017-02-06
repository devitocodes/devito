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

from sympy import (Eq, Indexed, IndexedBase, S,
                   collect, collect_const, cos, cse, flatten,
                   numbered_symbols, preorder_traversal, sin)

from devito.dimension import t, x, y, z
from devito.logger import dse, dse_warning

from devito.dse.extended_sympy import bhaskara_sin, bhaskara_cos
from devito.dse.graph import temporaries_graph
from devito.dse.inspection import (collect_aliases, estimate_cost, estimate_memory,
                                   is_binary_op, is_time_invariant, terminals)
from devito.dse.manipulation import flip_indices, rxreplace, unevaluate_arithmetic

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


def dse_transformation(func):

    def wrapper(self, state, **kwargs):
        if kwargs['mode'].intersection(set(self.triggers[func.__name__])):
            tic = time()
            state.update(**func(self, state))
            toc = time()

            key = '%s%d' % (func.__name__, len(self.timings))
            self.ops[key] = estimate_cost(state.exprs)
            self.timings[key] = toc - tic

    return wrapper


class State(object):

    def __init__(self, exprs):
        self.exprs = exprs
        self.mapper = OrderedDict()

    def update(self, exprs=None, mapper=None):
        self.exprs = exprs or self.exprs
        self.mapper = mapper or self.mapper

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
        Clusterize the expressions in ``self.exprs``. A cluster is an ordered
        collection of expressions that are necessary to compute a target expression
        (ie, an expression that is never read).

        Examples
        ========
        In the following list of expressions: ::

            temp1 = a*b
            temp2 = c
            temp3 = temp1 + temp2
            temp4 = temp2 + 5
            temp5 = d*e
            temp6 = f+g
            temp7 = temp5 + temp6

        There are three target expressions: temp3, temp4, temp7. There are therefore
        three clusters: ((temp1, temp2, temp3), (temp2, temp4), (temp5, temp6, temp7)).
        The first and second clusters share the expression temp2.
        """
        graph = temporaries_graph(self.exprs)
        targets = graph.targets
        clusters = [graph.trace(i.lhs) for i in targets]
        return clusters


class Rewriter(object):

    """
    Transform expressions to reduce their operation count.
    """

    triggers = {
        '_cse': ('basic', 'advanced'),
        '_factorize': ('factorize', 'advanced'),
        '_optimize_trigonometry': ('approx-trigonometry', 'advanced'),
        '_replace_time_invariants': ('glicm', 'advanced')
    }

    # Aggressive transformation if the operation count is greather than this
    # empirically determined threshold
    threshold = 15

    def __init__(self, exprs):
        self.exprs = exprs

        self.ops = OrderedDict([('baseline', estimate_cost(exprs))])
        self.timings = OrderedDict()

    def run(self, mode):
        state = State(self.exprs)

        self._cse(state, mode=mode)
        self._factorize(state, mode=mode)
        self._optimize_trigonometry(state, mode=mode)
        self._replace_time_invariants(state, mode=mode)
        self._factorize(state, mode=mode)

        self._finalize(state)

        self._summary(mode)

        return state

    @dse_transformation
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
            cost_expr = estimate_cost(expr)

            handle = collect_nested(expr)
            cost_handle = estimate_cost(handle)

            if cost_handle < cost_expr and cost_handle >= Rewriter.threshold:
                handle_prev = handle
                cost_prev = cost_expr
                while cost_handle < cost_prev:
                    handle_prev, handle = handle, collect_nested(handle)
                    cost_prev, cost_handle = cost_handle, estimate_cost(handle)
                cost_handle, handle = cost_prev, handle_prev

            processed.append(handle)

        return {'exprs': processed}

    @dse_transformation
    def _cse(self, state, **kwargs):
        """
        Perform common subexpression elimination.
        """

        temporaries, leaves = cse(state.exprs, numbered_symbols(_temp_prefix))
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
        leaves = rxreplace(leaves, mapper)
        kept = rxreplace([Eq(k, v) for k, v in keep.items()], mapper)

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

    @dse_transformation
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

    @dse_transformation
    def _replace_time_invariants(self, state, **kwargs):
        """
        Create a new expr' given expr where the longest time-invariant
        sub-expressions are replaced by temporaries. A mapper from the
        introduced temporaries to the corresponding time-invariant
        sub-expressions is also returned.

        Examples
        ========

        (a+b)*c[t] + s*d[t] + v*(d + e[t] + r)
            --> (t1*c[t] + s*d[t] + v*(e[t] + t2), {t1: (a+b), t2: (d+r)})
        (a*b[t] + c*d[t])*v[t]
            --> ((a*b[t] + c*d[t])*v[t], {})
        """

        template = "ti%d"
        graph = temporaries_graph(state.exprs)
        space_dimensions = graph.space_dimensions
        queue = graph.copy()

        # What expressions is it worth transforming (cm=cost model)?
        # Formula: ops(expr)*aliases(expr) > self.threshold <==> do it
        # For more information about "aliases", check out collect_aliases.__doc__
        aliases, clusters = collect_aliases([e.rhs for e in state.exprs])
        cm = lambda e: estimate_cost(e, True)*len(aliases.get(e, [e])) > self.threshold

        # Replace time invariants
        processed = []
        mapper = OrderedDict()
        while queue:
            k, v = queue.popitem(last=False)

            make = lambda m: Indexed(template % (len(m)+len(mapper)), *space_dimensions)
            invariant = lambda e: is_time_invariant(e, graph)
            handle, flag, mapped = replace_invariants(v, make, invariant, cm)

            if flag:
                mapper.update(mapped)
                for i in v.readby:
                    graph[i] = graph[i].construct({k: handle.rhs})
            else:
                processed.append(Eq(v.lhs, graph[v.lhs].rhs))

        # Squash aliases and tweak the affected indices accordingly
        reducible = OrderedDict()
        others = OrderedDict()
        for k, v in mapper.items():
            cluster = aliases.get(v)
            if cluster:
                index = clusters.index(cluster)
                reducible.setdefault(index, []).append(k)
            else:
                others[k] = v
        rule = {}
        reduced_mapper = OrderedDict()
        for i, cluster in enumerate(reducible.values()):
            for k in cluster:
                v, flipped = flip_indices(mapper[k], space_dimensions)
                assert len(flipped) == 1
                reduced_mapper[Indexed(template % i, *space_dimensions)] = v
                rule[k] = Indexed(template % i, *flipped.pop())
        handle, processed = list(processed), []
        for e in handle:
            processed.append(e.xreplace(rule))
        for k, v in others.items():
            reduced_mapper[k] = v.xreplace(rule)

        return {'exprs': processed, 'mapper': reduced_mapper}

    def _finalize(self, state):
        """
        Make sure that any subsequent sympy operation applied to the expressions
        in ``state.exprs`` does not alter the structure of the transformed objects.
        """
        exprs = [Eq(k, v) for k, v in state.mapper.items()] + state.exprs
        state.update(exprs=[unevaluate_arithmetic(e) for e in exprs])

    def _summary(self, mode):
        """
        Print a summary of the DSE transformations
        """

        if mode.intersection({'basic', 'advanced'}):
            try:
                # The state after CSE should be used as baseline for fairness
                baseline = self.ops['_cse0']
            except KeyError:
                baseline = self.ops['baseline']
            self.ops.pop('baseline')
            steps = " --> ".join("(%s) %d" % (filter(lambda c: not c.isdigit(), k), v)
                                 for k, v in self.ops.items())
            try:
                gain = float(baseline) / list(self.ops.values())[-1]
                summary = " %s flops; gain: %.2f X" % (steps, gain)
            except ZeroDivisionError:
                summary = ""
            elapsed = sum(self.timings.values())
            dse("%s [%.2f s]" % (summary, elapsed))


def collect_nested(expr):
    """
    Collect terms appearing in expr, checking all levels of the expression tree.

    :param expr: the expression to be factorized.
    """

    def run(expr):
        # Return semantic (rebuilt expression, factorization candidates)

        if expr.is_Float:
            return expr.func(*expr.atoms()), [expr]
        elif isinstance(expr, Indexed):
            return expr.func(*expr.args), []
        elif expr.is_Symbol:
            return expr.func(expr.name), [expr]
        elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
            return expr.func(), [expr]
        elif expr.is_Atom:
            return expr.func(*expr.atoms()), []
        elif expr.is_Add:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])

            w_numbers = [i for i in rebuilt if any(j.is_Number for j in i.args)]
            wo_numbers = [i for i in rebuilt if i not in w_numbers]

            w_numbers = collect_const(expr.func(*w_numbers))
            wo_numbers = expr.func(*wo_numbers)

            if wo_numbers:
                for i in flatten(candidates):
                    wo_numbers = collect(wo_numbers, i)

            rebuilt = expr.func(w_numbers, wo_numbers)
            return rebuilt, []
        elif expr.is_Mul:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])
            rebuilt = collect_const(expr.func(*rebuilt))
            return rebuilt, flatten(candidates)
        else:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])
            return expr.func(*rebuilt), flatten(candidates)

    return run(expr)[0]


def replace_invariants(expr, make, invariant=lambda e: e, cm=lambda e: True):
    """
    Replace all sub-expressions of ``expr`` such that ``invariant(expr) == True``
    with a temporary created through ``make(expr)``. A sub-expression ``e``
    within ``expr`` is not visited if ``cm(e) == False``.
    """

    def run(expr, root, mapper):
        # Return semantic: (rebuilt expr, True <==> invariant)

        if expr.is_Float:
            return expr.func(*expr.atoms()), True
        elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
            return expr.func(), True
        elif expr.is_Symbol:
            return expr.func(expr.name), invariant(expr)
        elif expr.is_Atom:
            return expr.func(*expr.atoms()), True
        elif isinstance(expr, Indexed):
            return expr.func(*expr.args), invariant(expr)
        elif expr.is_Equality:
            handle, flag = run(expr.rhs, expr.rhs, mapper)
            return expr.func(expr.lhs, handle, evaluate=False), flag
        else:
            children = [run(a, root, mapper) for a in expr.args]
            invs = [a for a, flag in children if flag]
            varying = [a for a, _ in children if a not in invs]
            if not invs:
                # Nothing is time-invariant
                return (expr.func(*varying, evaluate=False), False)
            elif len(invs) == len(children):
                # Everything is time-invariant
                if expr == root:
                    if cm(expr):
                        temporary = make(mapper)
                        mapper[temporary] = expr.func(*invs, evaluate=False)
                        return temporary, True
                    else:
                        return expr.func(*invs, evaluate=False), False
                else:
                    # Go look for longer expressions first
                    return expr.func(*invs, evaluate=False), True
            else:
                # Some children are time-invariant, but expr is time-dependent
                if cm(expr) and len(invs) > 1:
                    temporary = make(mapper)
                    mapper[temporary] = expr.func(*invs, evaluate=False)
                    return expr.func(*(varying + [temporary]), evaluate=False), False
                else:
                    return expr.func(*(varying + invs), evaluate=False), False

    mapper = OrderedDict()
    handle, flag = run(expr, expr, mapper)
    return handle, flag, mapper

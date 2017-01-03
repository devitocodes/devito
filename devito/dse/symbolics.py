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

from sympy import (Add, Eq, Indexed, IndexedBase, S,
                   collect, collect_const, cos, cse, flatten,
                   numbered_symbols, preorder_traversal, sin)

from devito.dimension import t, x, y, z
from devito.logger import dse, dse_warning

from devito.dse.extended_sympy import (bhaskara_sin, bhaskara_cos, unevaluate_arithmetic,
                                       flip_indices)
from devito.dse.graph import temporaries_graph
from devito.dse.inspection import (collect_aliases, estimate_cost,
                                   is_time_invariant, terminals)

__all__ = ['rewrite']

_temp_prefix = 'temp'


def rewrite(expr, mode='advanced'):
    """
    Transform expressions to reduce their operation count.

    :param expr: the target expression.
    :param mode: drive the expression transformation. Available modes are
                 'basic', 'factorize', 'approx-trigonometry' and 'advanced'
                 (default). They act as follows: ::

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
        return expr
    elif isinstance(mode, str):
        mode = set([mode])
    else:
        try:
            mode = set(mode)
        except TypeError:
            dse_warning("Arg mode must be str or tuple (got %s)" % type(mode))
            return expr
    if mode.isdisjoint({'basic', 'factorize', 'approx-trigonometry',
                        'glicm', 'advanced'}):
        dse_warning("Unknown rewrite mode(s) %s" % str(mode))
        return expr
    else:
        return Rewriter(expr).run(mode)


def dse_transformation(func):

    def wrapper(self, state, **kwargs):
        if kwargs['mode'].intersection(set(self.triggers[func.func_name])):
            tic = time()
            state.update(**func(self, state))
            toc = time()

            self.ops[func.func_name] = estimate_cost(state.exprs)
            self.timings[func.func_name] = toc - tic

    return wrapper


class State(object):

    def __init__(self, exprs):
        self.exprs = exprs
        self.mapper = OrderedDict()

    def update(self, exprs=None, mapper=None):
        self.exprs = exprs or self.exprs
        self.mapper = mapper or self.mapper


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

        self.ops = OrderedDict()
        self.timings = OrderedDict()

    def run(self, mode):
        state = State(self.exprs)

        self._cse(state, mode=mode)
        self._factorize(state, mode=mode)
        self._optimize_trigonometry(state, mode=mode)
        self._replace_time_invariants(state, mode=mode)

        self._finalize(state)

        self._summary(mode)

        return state.exprs

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

        temps, stencils = cse(state.exprs, numbered_symbols(_temp_prefix))

        # Restores the LHS
        for i in range(len(state.exprs)):
            stencils[i] = Eq(state.exprs[i].lhs, stencils[i].rhs)

        to_revert = {}
        to_keep = []

        # Restores IndexedBases if they are collected by CSE and
        # reverts changes to simple index operations (eg: t - 1)
        for temp, value in temps:
            if isinstance(value, IndexedBase):
                to_revert[temp] = value
            elif isinstance(value, Indexed):
                to_revert[temp] = value
            elif isinstance(value, Add) and not \
                    set([t, x, y, z]).isdisjoint(set(value.args)):
                to_revert[temp] = value
            else:
                to_keep.append((temp, value))

        # Restores the IndexedBases and the Indexes in the assignments to revert
        for temp, value in to_revert.items():
            s_dict = {}
            for arg in preorder_traversal(value):
                if isinstance(arg, Indexed):
                    new_indices = []
                    for index in arg.indices:
                        if index in to_revert:
                            new_indices.append(to_revert[index])
                        else:
                            new_indices.append(index)
                    if arg.base.label in to_revert:
                        s_dict[arg] = Indexed(to_revert[value.base.label], *new_indices)
            to_revert[temp] = value.xreplace(s_dict)

        subs_dict = {}

        # Builds a dictionary of the replacements
        for expr in stencils + [assign for temp, assign in to_keep]:
            for arg in preorder_traversal(expr):
                if isinstance(arg, Indexed):
                    new_indices = []
                    for index in arg.indices:
                        if index in to_revert:
                            new_indices.append(to_revert[index])
                        else:
                            new_indices.append(index)
                    if arg.base.label in to_revert:
                        subs_dict[arg] = Indexed(to_revert[arg.base.label], *new_indices)
                    elif tuple(new_indices) != arg.indices:
                        subs_dict[arg] = Indexed(arg.base, *new_indices)
                if arg in to_revert:
                    subs_dict[arg] = to_revert[arg]

        def recursive_replace(handle, subs_dict):
            replaced = []
            for i in handle:
                old, new = i, i.xreplace(subs_dict)
                while new != old:
                    old, new = new, new.xreplace(subs_dict)
                replaced.append(new)
            return replaced

        stencils = recursive_replace(stencils, subs_dict)
        to_keep = recursive_replace([Eq(temp, assign) for temp, assign in to_keep],
                                    subs_dict)

        # If the RHS of a temporary variable is the LHS of a stencil,
        # update the value of the temporary variable after the stencil
        new_stencils = []
        for stencil in stencils:
            new_stencils.append(stencil)
            for temp in to_keep:
                if stencil.lhs in preorder_traversal(temp.rhs):
                    new_stencils.append(temp)
                    break

        # Reshuffle to make sure temporaries come later than their read values
        processed = OrderedDict([(i.lhs, i) for i in to_keep + new_stencils])
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

            make_ti = lambda m: Indexed(template % (len(m)+len(mapper)), x, y, z)
            invariant = lambda e: is_time_invariant(e, graph)
            handle, flag, mapped = replace_invariants(v, make_ti, invariant, cm)

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
                v, flipped = flip_indices(mapper[k], (x, y, z))
                assert len(flipped) == 1
                reduced_mapper[Indexed(template % i, x, y, z)] = v
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

        summary = ""
        if mode.intersection({'basic', 'advanced'}):
            baseline = self.ops['_cse']
            steps = " --> ".join("(%s) %d" % (k, v) for k, v in self.ops.items())
            try:
                gain = float(baseline) / self.ops.values()[-1]
                summary = " %s flops; gain: %.2f X" % (steps, gain)
            except ZeroDivisionError:
                pass
        elapsed = sum(self.timings.values())
        dse("Rewriter:%s [%.2f s]" % (summary, elapsed))


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


def replace_invariants(expr, make_ti, invariant=lambda e: e, cm=lambda e: True):
    """
    Replace all sub-expressions of ``expr`` such that ``invariant(expr) == True``
    with a temporary created through ``make_ti(expr)``. A sub-expression ``e``
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
                        temporary = make_ti(mapper)
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
                    temporary = make_ti(mapper)
                    mapper[temporary] = expr.func(*invs, evaluate=False)
                    return expr.func(*(varying + [temporary]), evaluate=False), False
                else:
                    return expr.func(*(varying + invs), evaluate=False), False

    mapper = OrderedDict()
    handle, flag = run(expr, expr, mapper)
    return handle, flag, mapper

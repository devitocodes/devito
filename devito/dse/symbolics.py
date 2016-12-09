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

from sympy import (Add, Atom, Eq, Indexed, IndexedBase, S,
                   collect, collect_const, cos, cse, flatten,
                   numbered_symbols, preorder_traversal, sin)

from devito.dimension import t, x, y, z
from devito.logger import dse, dse_warning

from devito.dse.extended_sympy import taylor_sin, taylor_cos, unevaluate_arithmetic
from devito.dse.graph import temporaries_graph
from devito.dse.inspection import estimate_cost, terminals

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
                    * 'advanced': 'basic' + 'factorize' + 'approx-trigonometry'.
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
    if mode.isdisjoint({'basic', 'factorize', 'approx-trigonometry', 'advanced'}):
        dse_warning("Unknown rewrite mode(s) %s" % str(mode))
        return expr
    else:
        return Rewriter(expr).run(mode)


class Rewriter(object):

    """
    Transform expressions to reduce their operation count.
    """

    # Do more factorization sweeps if the expression operation count is
    # greater than this threshold
    FACTORIZER_THS = 15

    def __init__(self, exprs):
        self.exprs = exprs
        self.ops = OrderedDict()

    def run(self, mode):
        processed = self.exprs

        tic = time()

        if mode.intersection({'basic', 'advanced'}):
            processed = self._cse(processed)

        if mode.intersection({'factorize', 'advanced'}):
            processed = self._factorize(processed)

        if mode.intersection({'approx-trigonometry', 'advanced'}):
            processed = self._optimize_trigonometry(processed)

        if mode.intersection({'glicm', 'advanced'}):
            processed = self._replace_time_invariants(processed)

        processed = self._finalize(processed)

        toc = time()

        self._summary(mode, toc-tic)

        return processed

    def _factorize(self, exprs):
        """
        Collect terms in each expr in exprs based on the following heuristic:

            * Collect all literals;
            * Collect all temporaries produced by CSE;
            * If the expression has an operation count higher than
              self.FACTORIZER_THS, then this is applied recursively until
              no more factorization opportunities are available.
        """

        processed = []
        for expr in exprs:
            cost_expr = estimate_cost(expr)

            handle = collect_nested(expr)
            cost_handle = estimate_cost(handle)

            if cost_handle < cost_expr and cost_handle >= Rewriter.FACTORIZER_THS:
                handle_prev = handle
                cost_prev = cost_expr
                while cost_handle < cost_prev:
                    handle_prev, handle = handle, collect_nested(handle)
                    cost_prev, cost_handle = cost_handle, estimate_cost(handle)
                cost_handle, handle = cost_prev, handle_prev

            processed.append(handle)

        self.ops['factorizer'] = estimate_cost(processed)

        return processed

    def _cse(self, exprs):
        """
        Perform common subexpression elimination.
        """

        temps, stencils = cse(exprs, numbered_symbols(_temp_prefix))

        # Restores the LHS
        for i in range(len(exprs)):
            stencils[i] = Eq(exprs[i].lhs, stencils[i].rhs)

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

        processed = list(ordered.values())

        self.ops['cse'] = estimate_cost(processed)

        return processed

    def _optimize_trigonometry(self, exprs):
        """
        Rebuild ``exprs`` replacing trigonometric functions with Bhaskara
        polynomials.
        """

        processed = []
        for expr in exprs:
            handle = expr.replace(sin, taylor_sin)
            handle = handle.replace(cos, taylor_cos)
            processed.append(handle)

        self.ops['opt-trigo'] = estimate_cost(processed)

        return processed

    def _replace_time_invariants(self, exprs):
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

        graph = temporaries_graph(exprs)

        processed = []
        mapper = OrderedDict()
        while graph:
            k, v = graph.popitem(last=False)

            make_ti = lambda m: Indexed("ti%d" % (len(m)+len(mapper)), (x, y, z))
            is_invariant = lambda e: e not in graph or graph[e].is_time_invariant
            handle, flag, mapped = replace_invariants(v, t, make_ti, is_invariant)

            mapper.update(mapped)

            if flag:
                for i in v.readby:
                    graph[i] = Eq(i, graph[i].rhs.xreplace({k: handle.rhs}))
                graph = temporaries_graph(graph.values())
            else:
                processed.append(Eq(v.lhs, handle.rhs))

        self.ops['...'] = estimate_cost(processed)

        # TODO
        # 1) introduce decorator for: op count and timing.
        # 2) minimize temporaries by using
        #    - template (theta[i] aliases to theta[i+1]
        #    - full pusing if a temporary just lives out of time
        # 3) introduce the invariants in the generated code

        return processed, mapper

    def _finalize(self, exprs):
        """
        Make sure that any subsequent sympy operation applied to the expressions
        in ``exprs`` does not alter the structure of the transformed objects.
        """
        return [unevaluate_arithmetic(e) for e in exprs]

    def _summary(self, mode, elapsed):
        """
        Print a summary of the DSE transformations
        """

        if mode.intersection({'basic', 'advanced'}):
            baseline = self.ops['cse']
        else:
            summary = ""

        if baseline:
            steps = " --> ".join("(%s) %d" % (k, v) for k, v in self.ops.items())
            gain = float(baseline) / self.ops.values()[-1]
            summary = " %s flops; gain: %.2f X" % (steps, gain)

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


def replace_invariants(expr, dim, make_ti, is_invariant=lambda e: e):
    """
    Replace all sub-expressions of expr that are invariant in the dimension
    dim. Additional invariance rules can be set through the optional lambda
    function is_invariant.
    """

    def run(expr, root, mapper):
        # Return semantic: (rebuilt expr, time invariant flag)

        if expr.is_Float:
            return expr.func(*expr.atoms()), True
        elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
            return expr.func(), True
        elif expr.is_Symbol:
            return expr.func(expr.name), is_invariant(expr)
        elif expr.is_Atom:
            return expr.func(*expr.atoms()), True
        elif expr.is_Equality:
            handle, flag = run(expr.rhs, expr.rhs, mapper)
            return expr.func(expr.lhs, handle), flag
        elif isinstance(expr, Indexed):
            return expr.func(*expr.args), dim not in expr.atoms()
        else:
            children = [run(a, root, mapper) for a in expr.args]
            invs = [a for a, flag in children if flag]
            varying = [a for a, _ in children if a not in invs]
            if not invs:
                # Nothing is time-invariant
                return (expr.func(*varying), False)
            if len(invs) == len(children):
                # Everything is time-invariant
                if expr == root:
                    # Root is a special case
                    temporary = make_ti(mapper)
                    mapper[temporary] = expr.func(*expr.args)
                    return temporary, True
                else:
                    # Go look for longer expressions first
                    return expr.func(*invs), True
            else:
                # Some children are time-invariant, but expr is time-dependent
                if len(invs) == 1 and isinstance(invs[0], (Atom, Indexed)):
                    return expr.func(*(invs + varying)), False
                else:
                    temporary = make_ti(mapper)
                    mapper[temporary] = expr.func(*invs)
                    return expr.func(*(varying + [temporary])), False

    mapper = OrderedDict()
    handle, flag = run(expr, expr, mapper)
    return handle, flag, mapper

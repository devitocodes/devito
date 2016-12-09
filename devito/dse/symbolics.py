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

from sympy import (Add, Eq, Indexed, IndexedBase, S,
                   collect, collect_const, cos, cse, flatten,
                   numbered_symbols, preorder_traversal, sin)

from devito.dimension import t, x, y, z
from devito.logger import perfbad, perfok, warning

from devito.dse.extended_sympy import taylor_sin, taylor_cos
from devito.dse.inspection import estimate_cost, terminals, unevaluate_arithmetic

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
            warning("Arg mode must be a str or tuple (got %s instead)" % type(mode))
            return expr
    if mode.isdisjoint({'basic', 'factorize', 'approx-trigonometry', 'advanced'}):
        warning("Unknown rewrite mode(s) %s" % str(mode))
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

    def run(self, mode):
        processed = self.exprs

        if mode.intersection({'basic', 'advanced'}):
            processed = self._cse(processed)

        if mode.intersection({'factorize', 'advanced'}):
            processed = self._factorize(processed)

        if mode.intersection({'approx-trigonometry', 'advanced'}):
            processed = self._optimize_trigonometry(processed)

        processed = self._finalize(processed)

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
        cost_original, cost_processed = 1, 1
        for expr in exprs:
            handle = collect_nested(expr)

            cost_expr = estimate_cost(expr)
            cost_original += cost_expr

            cost_handle = estimate_cost(handle)

            if cost_handle < cost_expr and cost_handle >= Rewriter.FACTORIZER_THS:
                handle_prev = handle
                cost_prev = cost_expr
                while cost_handle < cost_prev:
                    handle_prev, handle = handle, collect_nested(handle)
                    cost_prev, cost_handle = cost_handle, estimate_cost(handle)
                cost_handle, handle = cost_prev, handle_prev

            processed.append(handle)
            cost_processed += cost_handle

        out = perfok if cost_processed < cost_original else perfbad
        out("Rewriter: %d --> %d flops (Gain: %.2f X)" %
            (cost_original, cost_processed, float(cost_original)/cost_processed))

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

        return list(ordered.values())

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

        return processed

    def _finalize(self, exprs):
        """
        Make sure that any subsequent sympy operation applied to the expressions
        in ``exprs`` does not alter the structure of the transformed objects.
        """
        return [unevaluate_arithmetic(e) for e in exprs]


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

"""
The Devito symbolic engine is built on top of SymPy and provides two
classes of functions:
- for inspection of expressions
- for (in-place) manipulation of expressions
- for creation of new objects given some expressions
All exposed functions are prefixed with 'dse' (devito symbolic engine)
"""

from __future__ import absolute_import

from collections import OrderedDict
from operator import itemgetter

import numpy as np
from sympy import (Add, Eq, Function, Indexed, IndexedBase, Symbol,
                   cse, lambdify, numbered_symbols,
                   preorder_traversal, symbols)

from devito.dimension import t, x, y, z
from devito.interfaces import SymbolicData
from devito.logger import warning, perfok, perfbad

__all__ = ['dse_dimensions', 'dse_symbols', 'dse_dtype', 'dse_indexify',
           'dse_cse', 'dse_tolambda']

_temp_prefix = 'temp'

# Inspection

def dse_dimensions(expr):
    """
    Collect all function dimensions used in a sympy expression.
    """
    dimensions = []

    for e in preorder_traversal(expr):
        if isinstance(e, SymbolicData):
            dimensions += [i for i in e.indices if i not in dimensions]

    return dimensions


def dse_symbols(expr):
    """
    Collect defined and undefined symbols used in a sympy expression.

    Defined symbols are functions that have an associated :class
    SymbolicData: object, or dimensions that are known to the devito
    engine. Undefined symbols are generic `sympy.Function` or
    `sympy.Symbol` objects that need to be substituted before generating
    operator C code.
    """
    defined = set()
    undefined = set()

    for e in preorder_traversal(expr):
        if isinstance(e, SymbolicData):
            defined.add(e.func(*e.indices))
        elif isinstance(e, Function):
            undefined.add(e)
        elif isinstance(e, Symbol):
            undefined.add(e)

    return list(defined), list(undefined)


def dse_dtype(expr):
    """
    Try to infer the data type of an expression.
    """
    dtypes = [e.dtype for e in preorder_traversal(expr) if hasattr(e, 'dtype')]
    return np.find_common_type(dtypes, [])


# Manipulation

def dse_indexify(expr):
    """
    Convert functions into indexed matrix accesses in sympy expression.

    :param expr: sympy function expression to be converted.
    """
    replacements = {}

    for e in preorder_traversal(expr):
        if hasattr(e, 'indexed'):
            replacements[e] = e.indexify()

    return expr.xreplace(replacements)


def dse_rewrite(expr, mode='basic'):
    """
    Transform expressions to create time-invariant computation.
    """

    if mode == True:
        return Rewriter(expr, mode='advanced').run()
    elif mode in ['basic', 'advanced']:
        return Rewriter(expr, mode=mode).run()
    else:
        warning("Illegal rewrite mode %s" % str(mode))
        return expr


class Rewriter(object):

    """
    Transform expressions to reduce their operation count.
    """

    # Do more factorization sweeps if the expression operation count is
    # greater than this threshold
    FACTORIZER_THS = 15

    def __init__(self, expr, mode='basic'):
        self.expr = expr
        self.mode = mode

    def run(self):
        processed = self.expr

        if self.mode in ['basic', 'advanced']:
            processed = self._cse()

        if self.mode in ['advanced']:
            processed = self._factorize(processed)

        return processed

    def _factorize(self, exprs, mode=None):
        """
        Collect terms in each expr in exprs based on the following heuristic:

            * Iff mode is 'aggressive', apply product expansion;
            * Collect all literals;
            * Collect all temporaries produced by CSE;
            * If the expression has an operation count higher than
              self.FACTORIZER_THS, then this is applied recursively until
              no more factorization opportunities are available.
        """
        if exprs is None:
            exprs = self.expr
        if not isinstance(exprs, list):
            exprs = [exprs]

        cost_original, cost_processed = 1, 1
        processed, expensive = [], []
        for expr in exprs:
            handle = expand_mul(expr) if mode == 'aggressive' else expr

            handle = collect_nested(handle)

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

    def _cse(self, exprs=None):
        """
        Perform common subexpression elimination.
        """
        if exprs is None:
            exprs = self.expr
        if not isinstance(exprs, list):
            exprs = [exprs]

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

        return ordered.values()


# Creation

def dse_tolambda(exprs):
    """
    Tranform an expression into a lambda.

    :param exprs: an expression or a list of expressions.
    """
    exprs = exprs if isinstance(exprs, list) else [exprs]

    lambdas = []

    for expr in exprs:
        terms = free_terms(expr.rhs)
        term_symbols = [symbols("i%d" % i) for i in range(len(terms))]

        # Substitute IndexedBase references to simple variables
        # lambdify doesn't support IndexedBase references in expressions
        tolambdify = expr.rhs.subs(dict(zip(terms, term_symbols)))
        lambdified = lambdify(term_symbols, tolambdify)
        lambdas.append((lambdified, terms))

    return lambdas


# Utilities

def free_terms(expr):
    """
    Find the free terms in an expression.
    """
    found = []

    for term in expr.args:
        if isinstance(term, Indexed):
            found.append(term)
        else:
            found += free_terms(term)

    return found


def terminals(expr, discard_indexed=False):
    indexed = list(expr.find(Indexed))

    # To be discarded
    junk = flatten(i.atoms() for i in indexed)

    symbols = list(expr.find(Symbol))
    symbols = [i for i in symbols if i not in junk]

    if discard_indexed:
        return set(symbols)
    else:
        return set(indexed + symbols)


def collect_nested(expr):
    """
    Collect terms appearing in expr, checking all levels of the expression tree.

    :param expr: the expression to be factorized.
    """

    def run(expr):
        # Return semantic (rebuilt expression, factorization candidates)

        if expr.is_Float:
            return expr.func(*expr.atoms()), [expr]
        elif expr.is_Symbol:
            return expr.func(expr.name), [expr]
        elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
            return expr.func(), [expr]
        elif expr.is_Atom:
            return expr.func(*expr.atoms()), []
        elif isinstance(expr, Indexed):
            return expr.func(*expr.args), []
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


def estimate_cost(handle):
    try:
        # Is it a plain SymPy object ?
        iter(handle)
    except TypeError:
        handle = [handle]
    try:
        # Is it a dict ?
        handle = handle.values()
    except AttributeError:
        try:
            # Must be a list of dicts then
            handle = flatten(i.values() for i in handle)
        except AttributeError:
            pass
    try:
        # At this point it must be a list of SymPy objects
        # We don't count non floating point operations
        handle = [i.rhs if i.is_Equality else i for i in handle]
        total_ops = sum(count_ops(i.args) for i in handle)
        non_flops = sum(count_ops(i.find(Indexed)) for i in handle)
        return total_ops - non_flops
    except:
        warning("Cannot estimate cost of %s" % str(handle))

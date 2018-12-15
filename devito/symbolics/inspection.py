from collections import Counter

from sympy import cos, sin

from devito.symbolics.search import retrieve_ops, search
from devito.logger import warning
from devito.tools import flatten

__all__ = ['count', 'estimate_cost']


def count(exprs, query):
    """
    Return a mapper ``{(k, v)}`` where ``k`` is a sub-expression in ``exprs``
    matching ``query`` and ``v`` is the number of its occurrences.
    """
    mapper = Counter()
    for expr in exprs:
        mapper.update(Counter(search(expr, query, 'all', 'bfs')))
    return dict(mapper)


def estimate_cost(expr, estimate_functions=False):
    """
    Estimate the operation count of an expression.

    Parameters
    ----------
    expr : expr-like or list of expr-like
        One or more expressions for which the operation count is calculated.
    estimate_functions : dict, optional
        A mapper from known functions (e.g., sin, cos) to (estimated) operation counts.
    """
    external_functions = {sin: 50, cos: 50}
    try:
        # Is it a plain SymPy object ?
        iter(expr)
    except TypeError:
        expr = [expr]
    try:
        # Is it a dict ?
        expr = expr.values()
    except AttributeError:
        try:
            # Must be a list of dicts then
            expr = flatten([i.values() for i in expr])
        except AttributeError:
            pass
    try:
        # At this point it must be a list of SymPy objects
        # We don't use SymPy's count_ops because we do not count integer arithmetic
        # (e.g., array index functions such as i+1 in A[i+1])
        # Also, the routine below is *much* faster than count_ops
        expr = [i.rhs if i.is_Equality else i for i in expr]
        operations = flatten(retrieve_ops(i) for i in expr)
        flops = 0
        for op in operations:
            if op.is_Function:
                if estimate_functions:
                    flops += external_functions.get(op.__class__, 1)
                else:
                    flops += 1
            else:
                flops += len(op.args) - (1 + sum(True for i in op.args if i.is_Integer))
        return flops
    except:
        warning("Cannot estimate cost of %s" % str(expr))

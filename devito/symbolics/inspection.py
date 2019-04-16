from collections import Counter

from sympy import cos, sin

from devito.symbolics.search import retrieve_ops, search
from devito.logger import warning
from devito.tools import as_tuple, flatten

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


def estimate_cost(exprs, estimate_functions=False):
    """
    Estimate the operation count of an expression.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        One or more expressions for which the operation count is calculated.
    estimate_functions : dict, optional
        A mapper from known functions (e.g., sin, cos) to (estimated) operation counts.
    """
    external_functions = {sin: 50, cos: 50}
    try:
        # Is it a plain symbol/array ?
        if exprs.is_AbstractFunction or exprs.is_AbstractSymbol:
            return 0
    except AttributeError:
        pass
    try:
        # Is it a dict ?
        exprs = exprs.values()
    except AttributeError:
        try:
            # Could still be a list of dicts
            exprs = flatten([i.values() for i in exprs])
        except (AttributeError, TypeError):
            pass
    try:
        # At this point it must be a list of SymPy objects
        # We don't use SymPy's count_ops because we do not count integer arithmetic
        # (e.g., array index functions such as i+1 in A[i+1])
        # Also, the routine below is *much* faster than count_ops
        exprs = [i.rhs if i.is_Equality else i for i in as_tuple(exprs)]
        operations = flatten(retrieve_ops(i) for i in exprs)
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
        warning("Cannot estimate cost of `%s`" % str(exprs))
        return 0

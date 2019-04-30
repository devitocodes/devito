from collections import Counter

from sympy import cos, sin, exp, log

from devito.symbolics.search import retrieve_xops, search
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


def estimate_cost(exprs, estimate=False):
    """
    Estimate the operation count of an expression.


    Parameters
    ----------
    exprs : expr-like or list of expr-like
        One or more expressions for which the operation count is calculated.
    estimate : bool, optional
        Defaults to False; if True, the following rules are applied:
            * Trascendental functions (e.g., cos, sin, ...) count as 50 ops.
            * Divisions (powers with a negative exponened) count as 25 ops.
    """
    trascendentals_cost = {sin: 50, cos: 50, exp: 50, log: 50}
    pow_cost = 50
    div_cost = 25

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
        operations = flatten(retrieve_xops(i) for i in exprs)
        flops = 0
        for op in operations:
            if op.is_Function:
                if estimate:
                    flops += trascendentals_cost.get(op.__class__, 1)
                else:
                    flops += 1
            elif op.is_Pow:
                if estimate:
                    if op.exp.is_Number:
                        if op.exp < 0:
                            flops += div_cost
                        elif op.exp == 0:
                            flops += 0
                        elif op.exp.is_Integer:
                            # Natural pows a**b are estimated as b-1 Muls
                            flops += op.exp - 1
                        else:
                            flops += pow_cost
                    else:
                        flops += pow_cost
                else:
                    flops += 1
            else:
                flops += len(op.args) - (1 + sum(True for i in op.args if i.is_Integer))
        return flops
    except:
        warning("Cannot estimate cost of `%s`" % str(exprs))
        return 0

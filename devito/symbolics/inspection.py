from collections import Counter
from operator import attrgetter

from sympy import Indexed, cos, sin

from devito.dimension import Dimension
from devito.symbolics.search import retrieve_indexed, retrieve_ops, search
from devito.symbolics.queries import q_timedimension
from devito.logger import warning
from devito.tools import flatten, filter_sorted, partial_order

__all__ = ['count', 'estimate_cost', 'estimate_memory', 'dimension_sort']


def count(exprs, query):
    """
    Return a mapper ``{(k, v)}`` where ``k`` is a sub-expression in ``exprs``
    matching ``query`` and ``v`` is the number of its occurrences.
    """
    mapper = Counter()
    for expr in exprs:
        mapper.update(Counter(search(expr, query, 'all', 'bfs')))
    return dict(mapper)


def estimate_cost(handle, estimate_functions=False):
    """Estimate the operation count of ``handle``.

    :param handle: a SymPy expression or an iterator of SymPy expressions.
    :param estimate_functions: approximate the operation count of known
                               functions (eg, sin, cos).
    """
    external_functions = {sin: 50, cos: 50}
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
            handle = flatten([i.values() for i in handle])
        except AttributeError:
            pass
    try:
        # At this point it must be a list of SymPy objects
        # We don't use SymPy's count_ops because we do not count integer arithmetic
        # (e.g., array index functions such as i+1 in A[i+1])
        # Also, the routine below is *much* faster than count_ops
        handle = [i.rhs if i.is_Equality else i for i in handle]
        operations = flatten(retrieve_ops(i) for i in handle)
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
        warning("Cannot estimate cost of %s" % str(handle))


def estimate_memory(handle, mode='realistic'):
    """
    Estimate the number of memory reads and writes.

    :param handle: a SymPy expression or an iterator of SymPy expressions.
    :param mode: Mode for computing the estimate:

    Estimate ``mode`` might be any of: ::

        * ideal: Also known as "compulsory traffic", which is the minimum
                 number of read/writes to be performed (ie, models an infinite cache).
        * ideal_with_stores: Like ideal, but a data item which is both read.
                             and written is counted twice (ie both load an
                             store are counted).
        * realistic: Assume that all datasets, even the time-independent ones,
                     need to be re-read at each time iteration.
    """
    assert mode in ['ideal', 'ideal_with_stores', 'realistic']

    def access(symbol):
        assert isinstance(symbol, Indexed)
        # Irregular accesses (eg A[B[i]]) are counted as compulsory traffic
        if any(i.atoms(Indexed) for i in symbol.indices):
            return symbol
        else:
            return symbol.base

    try:
        # Is it a plain SymPy object ?
        iter(handle)
    except TypeError:
        handle = [handle]

    if mode in ['ideal', 'ideal_with_stores']:
        filter = lambda s: any(q_timedimension(i) for i in s.atoms())
    else:
        filter = lambda s: s
    reads = set(flatten([retrieve_indexed(e.rhs) for e in handle]))
    writes = set(flatten([retrieve_indexed(e.lhs) for e in handle]))
    reads = set([access(s) for s in reads if filter(s)])
    writes = set([access(s) for s in writes if filter(s)])
    if mode == 'ideal':
        return len(set(reads) | set(writes))
    else:
        return len(reads) + len(writes)


def dimension_sort(expr, key=None):
    """
    Topologically sort the :class:`Dimension`s in ``expr``, based on the order
    in which they are encountered when visiting ``expr``.

    :param expr: The :class:`sympy.Eq` from which the :class:`Dimension`s are
                 extracted. They can appear both as array indices or as free
                 symbols.
    :param key: A callable used as key to enforce a final ordering.
    """
    # Get all Indexed dimensions, in the same order as the appear in /expr/
    constraints = []
    for i in retrieve_indexed(expr, mode='all'):
        constraint = []
        for ai, fi in zip(i.indices, i.base.function.indices):
            if ai.is_Number:
                constraint.append(fi)
            else:
                constraint.extend([d for d in ai.free_symbols
                                   if isinstance(d, Dimension) and d not in constraint])
        constraints.append(tuple(constraint))
    ordering = partial_order(constraints)

    # Add any leftover free dimensions (not an Indexed' index)
    dimensions = [i for i in expr.free_symbols if isinstance(i, Dimension)]
    dimensions = filter_sorted(dimensions, key=attrgetter('name'))  # for determinism
    ordering.extend([i for i in dimensions if i not in ordering])

    # Add parent dimensions
    derived = [i for i in ordering if i.is_Derived]
    for i in derived:
        ordering.insert(ordering.index(i), i.parent)

    return sorted(ordering, key=lambda i: not i.is_Time)

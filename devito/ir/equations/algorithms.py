from operator import attrgetter

from devito.dimension import Dimension
from devito.symbolics import retrieve_indexed
from devito.tools import filter_sorted, toposort

__all__ = ['dimension_sort']


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
    ordering = toposort(constraints)

    # Add any leftover free dimensions (not an Indexed' index)
    dimensions = [i for i in expr.free_symbols if isinstance(i, Dimension)]
    dimensions = filter_sorted(dimensions, key=attrgetter('name'))  # for determinism
    ordering.extend([i for i in dimensions if i not in ordering])

    # Add parent dimensions
    derived = [i for i in ordering if i.is_Derived]
    for i in derived:
        ordering.insert(ordering.index(i), i.parent)

    return sorted(ordering, key=lambda i: not i.is_Time)

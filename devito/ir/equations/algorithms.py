from operator import attrgetter

from devito.dimension import Dimension
from devito.symbolics import retrieve_indexed, split_affine
from devito.tools import filter_sorted, flatten, toposort

__all__ = ['dimension_sort']


def dimension_sort(expr, key=None):
    """
    Topologically sort the :class:`Dimension`s in ``expr``, based on the order
    in which they appear within :class:`Indexed`s.

    :param expr: The :class:`devito.Eq` from which the :class:`Dimension`s are
                 extracted.
    :param key: A callable used as key to enforce a final ordering.
    """

    def handle_indexed(indexed):
        constraint = []
        for i in indexed.indices:
            try:
                maybe_dim = split_affine(i).var
                if isinstance(maybe_dim, Dimension):
                    constraint.append(maybe_dim)
            except ValueError:
                # Maybe there are some nested Indexeds (e.g., the situation is A[B[i]])
                nested = flatten(handle_indexed(n) for n in retrieve_indexed(i))
                if nested:
                    constraint.extend(nested)
                else:
                    # Fallback: Just insert all the Dimensions we find, regardless of
                    # what the user is attempting to do
                    constraint.extend([d for d in filter_sorted(i.free_symbols)
                                       if isinstance(d, Dimension)])
        return constraint

    constraints = [handle_indexed(i) for i in retrieve_indexed(expr, mode='all')]

    ordering = toposort(constraints)

    # Add in leftover free dimensions (not an Indexed' index)
    extra = set([i for i in expr.free_symbols if isinstance(i, Dimension)])

    # Add in pure data dimensions (e.g., those accessed only via explicit values,
    # such as A[3])
    indexeds = retrieve_indexed(expr, deep=True)
    if indexeds:
        extra.update(set.union(*[set(i.function.indices) for i in indexeds]))

    # Enforce determinism
    extra = filter_sorted(extra, key=attrgetter('name'))

    ordering.extend([i for i in extra if i not in ordering])

    # Add in parent dimensions
    for i in list(ordering):
        if i.is_Derived and i.parent not in ordering:
            ordering.insert(ordering.index(i), i.parent)

    return sorted(ordering, key=key)

from operator import attrgetter

from devito.symbolics import retrieve_indexed, split_affine
from devito.tools import PartialOrderTuple, filter_sorted, flatten
from devito.types import Dimension

__all__ = ['dimension_sort']


def dimension_sort(expr):
    """
    Topologically sort the :class:`Dimension`s in ``expr``, based on the order
    in which they appear within :class:`Indexed`s.
    """

    def handle_indexed(indexed):
        relation = []
        for i in indexed.indices:
            try:
                maybe_dim = split_affine(i).var
                if isinstance(maybe_dim, Dimension):
                    relation.append(maybe_dim)
            except ValueError:
                # Maybe there are some nested Indexeds (e.g., the situation is A[B[i]])
                nested = flatten(handle_indexed(n) for n in retrieve_indexed(i))
                if nested:
                    relation.extend(nested)
                else:
                    # Fallback: Just insert all the Dimensions we find, regardless of
                    # what the user is attempting to do
                    relation.extend([d for d in filter_sorted(i.free_symbols)
                                     if isinstance(d, Dimension)])
        return tuple(relation)

    relations = {handle_indexed(i) for i in retrieve_indexed(expr, mode='all')}

    # Add in any implicit dimension (typical of scalar temporaries, or Step)
    relations.add(expr.implicit_dims)

    # Add in leftover free dimensions (not an Indexed' index)
    extra = set([i for i in expr.free_symbols if isinstance(i, Dimension)])

    # Add in pure data dimensions (e.g., those accessed only via explicit values,
    # such as A[3])
    indexeds = retrieve_indexed(expr, deep=True)
    extra.update(set().union(*[set(i.function.indices) for i in indexeds]))

    # Enforce determinism
    extra = filter_sorted(extra, key=attrgetter('name'))

    # Add in implicit relations for parent dimensions
    # -----------------------------------------------
    # 1) Note that (d.parent, d) is what we want, while (d, d.parent) would be
    # wrong; for example, in `((t, time), (t, x, y), (x, y))`, `x` could now
    # preceed `time`, while `t`, and therefore `time`, *must* appear before `x`,
    # as indicated by the second relation
    implicit_relations = {(d.parent, d) for d in extra if d.is_Derived}
    # 2) To handle cases such as `((time, xi), (x,))`, where `xi` a SubDimension
    # of `x`, besides `(x, xi)`, we also have to add `(time, x)` so that we
    # obtain the desired ordering `(time, x, xi)`. W/o `(time, x)`, the ordering
    # `(x, time, xi)` might be returned instead, which would be non-sense
    implicit_relations.update({tuple(d.root for d in i) for i in relations})

    ordering = PartialOrderTuple(extra, relations=(relations | implicit_relations))

    return ordering

from devito.ir.iet import Expression, Iteration, List, MergeOuterIterations
from devito.symbolics import Eq
from devito.tools import flatten

__all__ = ['compose_nodes', 'copy_arrays']


def compose_nodes(nodes, retrieve=False):
    """
    Build an Iteration/Expression tree by nesting the nodes in ``nodes``.
    """
    l = list(nodes)
    tree = []

    if not isinstance(l[0], Iteration):
        # Nothing to compose
        body = flatten(l)
        body = List(body=body) if len(body) > 1 else body[0]
    else:
        body = l.pop(-1)
        while l:
            handle = l.pop(-1)
            body = handle._rebuild(body, **handle.args_frozen)
            tree.append(body)

    if retrieve is True:
        tree = list(reversed(tree))
        return body, tree
    else:
        return body


def copy_arrays(mapper, reverse=False):
    """
    Build an Iteration/Expression tree performing the copy ``k = v``, or
    ``v = k`` if reverse=True, for each (k, v) in mapper. (k, v) are expected
    to be of type :class:`IndexedData`. The loop bounds are inferred from
    the dimensions used in ``k``.
    """
    if not mapper:
        return ()

    # Build the Iteration tree for the copy
    iterations = []
    for k, v in mapper.items():
        handle = []
        indices = k.function.indices
        for i, j in zip(k.shape, indices):
            handle.append(Iteration([], dimension=j, limits=i))
        lhs, rhs = (v, k) if reverse else (k, v)
        handle.append(Expression(Eq(lhs[indices], rhs[indices]), dtype=k.function.dtype))
        iterations.append(compose_nodes(handle))

    # Maybe some Iterations are mergeable
    iterations = MergeOuterIterations().visit(iterations)

    return iterations

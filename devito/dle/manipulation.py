from sympy import Eq

from devito.cgen_utils import ccode
from devito.nodes import Expression, Iteration
from devito.visitors import MergeOuterIterations

__all__ = ['compose_nodes', 'copy_arrays']


def compose_nodes(nodes):
    """Build an Iteration/Expression tree by nesting the nodes in ``nodes``."""
    l = list(nodes)

    body = l.pop(-1)
    while l:
        handle = l.pop(-1)
        body = handle._rebuild(body, **handle.args_frozen)

    return body


def copy_arrays(mapper):
    """Build an Iteration/Expression tree performing the copy ``k = v`` for each
    (k, v) in mapper. (k, v) are expected to be of type :class:`IndexedData`."""

    # Build the Iteration tree for the copy
    iterations = []
    for k, v in mapper.items():
        handle = []
        indices = k.function.indices
        for i, j in zip(k.shape, indices):
            handle.append(Iteration([], dimension=j, limits=j.symbolic_size))
        handle.append(Expression(Eq(k[indices], v[indices]), dtype=k.function.dtype))
        iterations.append(compose_nodes(handle))

    # Maybe some Iterations are mergeable
    iterations = MergeOuterIterations().visit(iterations)

    return iterations

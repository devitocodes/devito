from devito.ir.iet import Expression, Iteration, List, FindSections, MergeOuterIterations
from devito.symbolics import Eq
from devito.tools import as_tuple, flatten

__all__ = ['filter_iterations', 'retrieve_iteration_tree', 'is_foldable',
           'compose_nodes', 'copy_arrays']


def retrieve_iteration_tree(node, mode='normal'):
    """Return a list of all :class:`Iteration` sub-trees rooted in ``node``.
    For example, given the Iteration tree:

        .. code-block:: c

           Iteration i
             expr0
             Iteration j
               Iteraion k
                 expr1
             Iteration p
               expr2

    Return the list: ::

        [(Iteration i, Iteration j, Iteration k), (Iteration i, Iteration p)]

    :param node: The searched Iteration/Expression tree.
    :param mode: Accepted values are 'normal' (default) and 'superset', in which
                 case iteration trees that are subset of larger iteration trees
                 are dropped.
    """
    assert mode in ('normal', 'superset')

    trees = [i for i in FindSections().visit(node) if i]
    if mode == 'normal':
        return trees
    else:
        match = []
        for i in trees:
            if any(set(i).issubset(set(j)) for j in trees if i != j):
                continue
            match.append(i)
        return match


def filter_iterations(tree, key=lambda i: i, stop=lambda: False):
    """
    Given an iterable of :class:`Iteration` objects, return a new list
    containing all items such that ``key(o)`` is True.

    This function accepts an optional argument ``stop``. This may be either a
    lambda function, specifying a stop criterium, or any of the following
    special keywords: ::

        * 'any': Return as soon as ``key(o)`` is False and at least one
                 item has been collected.
        * 'asap': Return as soon as at least one item has been collected and
                  all items for which ``key(o)`` is False have been encountered.

    It is useful to specify a ``stop`` criterium when one is searching the
    first Iteration in an Iteration/Expression tree for which a given property
    does not hold.
    """
    assert callable(stop) or stop in ['any', 'asap']

    tree = list(tree)
    filtered = []
    off = []

    if stop == 'any':
        stop = lambda: len(filtered) > 0
    elif stop == 'asap':
        hits = [i for i in tree if not key(i)]
        stop = lambda: len(filtered) > 0 and len(off) == len(hits)

    for i in tree:
        if key(i):
            filtered.append(i)
        else:
            off.append(i)
        if stop():
            break

    return filtered


def is_foldable(nodes):
    """
    Return True if the iterable ``nodes`` consists of foldable :class:`Iteration`
    objects, False otherwise.
    """
    nodes = as_tuple(nodes)
    if len(nodes) <= 1 or any(not i.is_Iteration for i in nodes):
        return False
    main = nodes[0]
    return all(i.dim == main.dim and i.limits == main.limits and i.index == main.index
               and i.properties == main.properties for i in nodes)


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

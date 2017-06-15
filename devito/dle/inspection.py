from devito.visitors import FindSections
from devito.tools import as_tuple

__all__ = ['filter_iterations', 'retrieve_iteration_tree', 'is_foldable']


def retrieve_iteration_tree(node, mode='normal'):
    """Return a list of all :class:`Iteration` sub-trees rooted in ``node``.
    For example, given the Iteration tree:

        .. code-block::
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
                 case iteration trees that are subset of larget iteration trees
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

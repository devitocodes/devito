from devito.visitors import FindSections

__all__ = ['filter_iterations', 'retrieve_iteration_tree']


def retrieve_iteration_tree(node):
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
    """

    return [i for i in FindSections().visit(node).keys() if i]


def filter_iterations(tree, key=lambda i: i, stop=lambda i: False):
    """
    Given an iterable of :class:`Iteration` objects, return a new list
    containing all items such that ``key(o)`` is True.

    This function accepts an optional argument ``stop``. This may be either a
    lambda function, specifying a stop criterium, or the special keyword
    'consecutive', which makes the function return as soon as ``key(o)``
    gives False and at least one item has been collected.
    """

    filtered = []

    if stop == 'consecutive':
        stop = lambda: len(filtered) > 0

    for i in tree:
        if key(i):
            filtered.append(i)
        elif stop():
            break

    return filtered

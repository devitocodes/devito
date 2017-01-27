from collections import OrderedDict

from devito.visitors import FindSections

__all__ = ['retrieve_iteration_tree']


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

    return FindSections().visit(node).keys()

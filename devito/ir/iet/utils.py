from devito.ir.iet import Iteration, List, IterationTree, FindSections, FindSymbols
from devito.symbolics import Macro
from devito.tools import flatten
from devito.types import Array, LocalObject

__all__ = ['filter_iterations', 'retrieve_iteration_tree',
           'compose_nodes', 'derive_parameters']


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

    trees = [IterationTree(i) for i in FindSections().visit(node) if i]
    if mode == 'normal':
        return trees
    else:
        match = []
        for i in trees:
            if any(set(i).issubset(set(j)) for j in trees if i != j):
                continue
            match.append(i)
        return IterationTree(match)


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


def derive_parameters(nodes, drop_locals=False):
    """
    Derive all input parameters (function call arguments) from an IET
    by collecting all symbols not defined in the tree itself.
    """
    # Pick all free symbols and symbolic functions from the kernel
    functions = FindSymbols('symbolics').visit(nodes)
    free_symbols = FindSymbols('free-symbols').visit(nodes)

    # Filter out function base symbols and use real function objects
    function_names = [s.name for s in functions]
    symbols = [s for s in free_symbols if s.name not in function_names]
    symbols = functions + symbols

    defines = [s.name for s in FindSymbols('defines').visit(nodes)]
    parameters = tuple(s for s in symbols if s.name not in defines)

    # Drop globally-visible objects
    parameters = [p for p in parameters if not isinstance(p, Macro)]

    # Filter out locally-allocated Arrays and Objects
    if drop_locals:
        parameters = [p for p in parameters
                      if not (isinstance(p, Array) and (p._mem_heap or p._mem_stack))]
        parameters = [p for p in parameters if not isinstance(p, LocalObject)]

    return parameters

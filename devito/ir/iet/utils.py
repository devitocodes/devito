from collections import OrderedDict

from devito.ir.iet import (Iteration, List, IterationTree, FindSections, FindSymbols,
                           FindNodes, Section, Expression)
from devito.symbolics import Literal, Macro
from devito.tools import flatten, ReducerMap
from devito.types import Array, LocalObject

__all__ = ['filter_iterations', 'retrieve_iteration_tree',
           'compose_nodes', 'derive_parameters', 'find_affine_trees']


def retrieve_iteration_tree(node, mode='normal'):
    """
    A list with all :class:`Iteration` sub-trees within an IET.

    Examples
    --------
    Given the Iteration tree:

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

    Parameters
    ----------
    iet : Node
        The searched Iteration/Expression tree.
    mode : str, optional
        - ``normal``
        - ``superset``: Iteration trees that are subset of larger iteration trees
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


def filter_iterations(tree, key=lambda i: i):
    """
    Return the first sub-sequence of consecutive Iterations such that
    ``key(iteration)`` is True.
    """
    filtered = []
    for i in tree:
        if key(i):
            filtered.append(i)
        elif len(filtered) > 0:
            break
    return filtered


def compose_nodes(nodes, retrieve=False):
    """Build an IET by nesting ``nodes``."""
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
    parameters = [p for p in parameters if not isinstance(p, (Literal, Macro))]

    # Filter out locally-allocated Arrays and Objects
    if drop_locals:
        parameters = [p for p in parameters
                      if not (isinstance(p, Array) and (p._mem_heap or p._mem_stack))]
        parameters = [p for p in parameters if not isinstance(p, LocalObject)]

    return parameters


def find_affine_trees(iet):
    """
    Find affine trees. A tree is affine when all of the array accesses are
    constant/affine functions of the Iteration variables and the Iteration bounds
    are fixed (but possibly symbolic).

    Parameters
    ----------
    iet : `Node`
        The searched tree

    Returns
    -------
    list of `Node`
        Each item in the list is the root of an affine tree
    """
    affine = OrderedDict()
    roots = [i for i in FindNodes(Iteration).visit(iet) if i.dim.is_Time]
    for root in roots:
        sections = FindNodes(Section).visit(root)
        for section in sections:
            for tree in retrieve_iteration_tree(section):
                if not all(i.is_Affine for i in tree):
                    # Non-affine array accesses not supported
                    break
                exprs = [i.expr for i in FindNodes(Expression).visit(tree.root)]
                grid = ReducerMap([('', i.grid) for i in exprs if i.grid]).unique('')
                writeto_dimensions = tuple(i.dim.root for i in tree)
                if grid.dimensions == writeto_dimensions:
                    affine.setdefault(section, []).append(tree)
                else:
                    break
    return affine

from collections.abc import Callable, Iterable, Iterator
from itertools import chain
from typing import Any, Literal

import numpy as np
import sympy

from devito.symbolics.queries import (q_indexed, q_function, q_terminal, q_leaf,
                                      q_symbol, q_dimension, q_derivative)
from devito.tools import as_tuple

__all__ = ['retrieve_indexed', 'retrieve_functions', 'retrieve_function_carriers',
           'retrieve_terminals', 'retrieve_symbols', 'retrieve_dimensions',
           'retrieve_derivatives', 'search']


Expression = sympy.Basic | np.number | int | float


class List(list[Expression]):
    """
    A list that aliases `extend` to `update` to mirror the `set` interface.
    """

    def update(self, obj: Iterable[Expression]) -> None:
        self.extend(obj)


Mode = Literal['all', 'unique']
modes: dict[Mode, type[List] | type[set[Expression]]] = {
    'all': List,
    'unique': set
}


class Search:
    def __init__(self, query: Callable[[Expression], bool], deep: bool = False) -> None:
        """
        Search objects in an expression. This is much quicker than the more general
        SymPy's find.

        Parameters
        ----------
        query
            Any query from :mod:`queries`.
        deep : bool, optional
            If True, propagate the search within an Indexed's indices. Defaults to False.
        """
        self.query = query
        self.deep = deep

    def _next(self, expr: Expression) -> Iterable[Expression]:
        if self.deep and expr.is_Indexed:
            return expr.indices
        elif q_leaf(expr):
            return ()
        return expr.args

    def visit_postorder(self, expr: Expression) -> Iterator[Expression]:
        """
        Visit the expression with a postorder traversal, yielding all hits.
        """
        for i in self._next(expr):
            yield from self.visit_postorder(i)
        if self.query(expr):
            yield expr

    def visit_preorder(self, expr: Expression) -> Iterator[Expression]:
        """
        Visit the expression with a preorder traversal, yielding all hits.
        """
        if self.query(expr):
            yield expr
        for i in self._next(expr):
            yield from self.visit_preorder(i)

    def visit_preorder_first_hit(self, expr: Expression) -> Iterator[Expression]:
        """
        Visit the expression in preorder and return a tuple containing the first hit,
        if any. This can return more than a single result, as it looks for the first
        hit from any branch but may find a hit in multiple branches.
        """
        if self.query(expr):
            yield expr
            return
        for i in self._next(expr):
            yield from self.visit_preorder_first_hit(i)


def search(exprs: Expression | Iterable[Expression],
           query: type | Callable[[Any], bool],
           mode: Mode = 'unique',
           visit: Literal['dfs', 'bfs', 'bfs_first_hit'] = 'dfs',
           deep: bool = False) -> List | set[Expression]:
    """Interface to Search."""

    assert mode in ('all', 'unique'), "Unknown mode"

    if isinstance(query, type):
        Q = lambda obj: isinstance(obj, query)
    else:
        Q = query

    # Search doesn't actually use a BFS (rather, a preorder DFS), but the terminology
    # is retained in this function's parameters for backwards compatibility
    searcher = Search(Q, deep)
    match visit:
        case 'dfs':
            _search = searcher.visit_postorder
        case 'bfs':
            _search = searcher.visit_preorder
        case 'bfs_first_hit':
            _search = searcher.visit_preorder_first_hit
        case _:
            raise ValueError(f"Unknown visit mode '{visit}'")

    exprs = filter(lambda e: isinstance(e, sympy.Basic), as_tuple(exprs))
    found = modes[mode](chain(*map(_search, exprs)))

    return found


# Shorthands


def retrieve_indexed(exprs, mode='all', deep=False):
    """Shorthand to retrieve the Indexeds in ``exprs``."""
    return search(exprs, q_indexed, mode, 'dfs', deep)


def retrieve_functions(exprs, mode='all', deep=False):
    """Shorthand to retrieve the DiscreteFunctions in `exprs`."""
    indexeds = search(exprs, q_indexed, mode, 'dfs', deep)

    functions = search(exprs, q_function, mode, 'dfs', deep)
    functions.update({i.function for i in indexeds})

    return functions


def retrieve_symbols(exprs, mode='all'):
    """Shorthand to retrieve the Scalar in `exprs`."""
    return search(exprs, q_symbol, mode, 'dfs')


def retrieve_function_carriers(exprs, mode='all'):
    """
    Shorthand to retrieve the DiscreteFunction carriers in ``exprs``. An
    object carries a DiscreteFunction if any of the following conditions are met: ::

        * it is itself a DiscreteFunction, OR
        * it is an Indexed, which internally has a pointer to a DiscreteFunction.
    """
    query = lambda i: q_function(i) or q_indexed(i)
    retval = search(exprs, query, mode, 'dfs')
    # Filter off Indexeds not carrying a DiscreteFunction
    for i in list(retval):
        try:
            i.function
        except AttributeError:
            retval.remove(i)
    return retval


def retrieve_terminals(exprs, mode='all', deep=False):
    """Shorthand to retrieve Indexeds and Symbols within ``exprs``."""
    return search(exprs, q_terminal, mode, 'dfs', deep)


def retrieve_dimensions(exprs, mode='all', deep=False):
    """Shorthand to retrieve the dimensions in ``exprs``."""
    return search(exprs, q_dimension, mode, 'dfs', deep)


def retrieve_derivatives(exprs, mode='all', deep=False):
    """Shorthand to retrieve the Derivatives in ``exprs``."""
    return search(exprs, q_derivative, mode, 'dfs', deep)

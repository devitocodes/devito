from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal

import sympy

from devito.symbolics.queries import (q_indexed, q_function, q_terminal, q_leaf,
                                      q_symbol, q_dimension, q_derivative)
from devito.tools import as_tuple

__all__ = ['retrieve_indexed', 'retrieve_functions', 'retrieve_function_carriers',
           'retrieve_terminals', 'retrieve_symbols', 'retrieve_dimensions',
           'retrieve_derivatives', 'search']


class Set(set):

    @staticmethod
    def wrap(obj) -> set:
        return {obj}


class List(list):

    @staticmethod
    def wrap(obj) -> list:
        return [obj]

    def update(self, obj: Iterable[Any]) -> None:
        return self.extend(obj)
    

modes: dict[Literal['all', 'unique'], type[List] | type[Set]] = {
    'all': List,
    'unique': Set
}


class Search:
    def __init__(self, query: Callable[[Any], bool], deep: bool = False) -> None:
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

    def _next(self, expr) -> Iterator[Any]:
        if self.deep and expr.is_Indexed:
            yield from expr.indices
        elif not q_leaf(expr):
            yield from expr.args

    def visit_postorder(self, expr) -> Iterator[Any]:
        for i in self._next(expr):
            yield from self.visit_postorder(i)
        if self.query(expr):
            yield expr

    def visit_preorder(self, expr) -> Iterator[Any]:
        if self.query(expr):
            yield expr
        for i in self._next(expr):
            yield from self.visit_preorder(i)

    def visit_preorder_first_hit(self, expr) -> tuple[Any, ...]:
        """Visit the expression in preorder and return the first hit."""
        if self.query(expr):
            return (expr,)
        for i in self._next(expr):
            result = self.visit_preorder_first_hit(i)
            if result:
                return result
        return ()



def search(exprs,
           query: type | Callable[[Any], bool],
           mode: Literal['all', 'unique'] = 'unique',
           visit: Literal['dfs', 'bfs', 'bfs_first_hit'] = 'dfs',
           deep: bool = False) -> List | Set:
    """Interface to Search."""

    assert mode in ('all', 'unique'), "Unknown mode"

    if isinstance(query, type):
        Q = lambda obj: isinstance(obj, query)
    else:
        Q = query

    # Search doesn't actually use a BFS (rather, a preorder DFS), but the terminology
    # is retained in this function's parameters for backwards compatibility
    searcher = Search(Q, deep)

    if visit == 'dfs':
        _visit = searcher.visit_postorder
    elif visit == 'bfs':
        _visit = searcher.visit_preorder
    elif visit == 'bfs_first_hit':
        _visit = searcher.visit_preorder_first_hit
    else:
        raise ValueError(f"Unknown visit mode '{visit}'")

    found = modes[mode]()
    for e in as_tuple(exprs):
        if not isinstance(e, sympy.Basic):
            continue

        found.update(_visit(e))

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
    """Shorthand to retrieve the Scalar in ``exprs``."""
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

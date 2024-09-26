import sympy

from devito.symbolics.queries import (q_indexed, q_function, q_terminal, q_leaf,
                                      q_symbol, q_dimension, q_derivative)
from devito.tools import as_tuple

__all__ = ['retrieve_indexed', 'retrieve_functions', 'retrieve_function_carriers',
           'retrieve_terminals', 'retrieve_symbols', 'retrieve_dimensions',
           'retrieve_derivatives', 'search']


class Search:

    class Set(set):

        @staticmethod
        def wrap(obj):
            return {obj}

    class List(list):

        @staticmethod
        def wrap(obj):
            return [obj]

        def update(self, obj):
            return self.extend(obj)

    modes = {
        'unique': Set,
        'all': List
    }

    def __init__(self, query, mode, deep=False):
        """
        Search objects in an expression. This is much quicker than the more
        general SymPy's find.

        Parameters
        ----------
        query
            Any query from :mod:`queries`.
        mode : str
            Either 'unique' or 'all' (catch all instances).
        deep : bool, optional
            If True, propagate the search within an Indexed's indices. Defaults to False.
        """
        self.query = query
        self.collection = self.modes[mode]
        self.deep = deep

    def _next(self, expr):
        if self.deep and expr.is_Indexed:
            return expr.indices
        elif q_leaf(expr):
            return ()
        else:
            return expr.args

    def dfs(self, expr):
        """
        Perform a DFS search.

        Parameters
        ----------
        expr : expr-like
            The searched expression.
        """
        found = self.collection()
        for a in self._next(expr):
            found.update(self.dfs(a))
        if self.query(expr):
            found.update(self.collection.wrap(expr))
        return found

    def bfs(self, expr):
        """
        Perform a BFS search.

        Parameters
        ----------
        expr : expr-like
            The searched expression.
        """
        found = self.collection()
        if self.query(expr):
            found.update(self.collection.wrap(expr))
        for a in self._next(expr):
            found.update(self.bfs(a))
        return found

    def bfs_first_hit(self, expr):
        """
        Perform a BFS search, returning immediately when a node matches the query.

        Parameters
        ----------
        expr : expr-like
            The searched expression.
        """
        found = self.collection()
        if self.query(expr):
            found.update(self.collection.wrap(expr))
            return found
        for a in self._next(expr):
            found.update(self.bfs_first_hit(a))
        return found


def search(exprs, query, mode='unique', visit='dfs', deep=False):
    """Interface to Search."""

    assert mode in Search.modes, "Unknown mode"

    if isinstance(query, type):
        Q = lambda obj: isinstance(obj, query)
    else:
        Q = query

    searcher = Search(Q, mode, deep)

    found = Search.modes[mode]()
    for e in as_tuple(exprs):
        if not isinstance(e, sympy.Basic):
            continue

        if visit == 'dfs':
            found.update(searcher.dfs(e))
        elif visit == 'bfs':
            found.update(searcher.bfs(e))
        elif visit == "bfs_first_hit":
            found.update(searcher.bfs_first_hit(e))
        else:
            raise ValueError("Unknown visit type `%s`" % visit)

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

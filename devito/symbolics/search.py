from devito.symbolics.queries import (q_indexed, q_function, q_terminal,
                                      q_leaf, q_op, q_trigonometry)

__all__ = ['retrieve_indexed', 'retrieve_functions', 'retrieve_function_carriers',
           'retrieve_terminals', 'retrieve_ops', 'retrieve_trigonometry', 'search']


class Search(object):

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
        if self.deep is True and expr.is_Indexed:
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


def search(expr, query, mode='unique', visit='dfs', deep=False):
    """Interface to Search."""

    assert mode in Search.modes, "Unknown mode"
    assert visit in ['dfs', 'bfs', 'bfs_first_hit'], "Unknown visit type"

    searcher = Search(query, mode, deep)
    if visit == 'dfs':
        return searcher.dfs(expr)
    elif visit == 'bfs':
        return searcher.bfs(expr)
    else:
        return searcher.bfs_first_hit(expr)


# Shorthands


def retrieve_indexed(expr, mode='unique', deep=False):
    """Shorthand to retrieve the Indexeds in ``expr``."""
    return search(expr, q_indexed, mode, 'dfs', deep)


def retrieve_functions(expr, mode='unique'):
    """Shorthand to retrieve the DiscretizedFunctions in ``expr``."""
    return search(expr, q_function, mode, 'dfs')


def retrieve_function_carriers(expr, mode='unique'):
    """
    Shorthand to retrieve the DiscretizedFunction carriers in ``expr``. An
    object carreis a DiscretizedFunction if any of the following conditions are met: ::

        * it is itself a DiscretizedFunction, OR
        * it is an Indexed, which internally has a pointer to a DiscretizedFunction.
    """
    query = lambda i: q_function(i) or q_indexed(i)
    retval = search(expr, query, mode, 'dfs')
    # Filter off Indexeds not carrying a DiscretizedFunction
    for i in list(retval):
        try:
            i.function
        except AttributeError:
            retval.remove(i)
    return retval


def retrieve_terminals(expr, mode='unique', deep=False):
    """Shorthand to retrieve Indexeds and Symbols within ``expr``."""
    return search(expr, q_terminal, mode, 'dfs', deep)


def retrieve_trigonometry(expr):
    """Shorthand to retrieve the trigonometric functions within ``expr``."""
    return search(expr, q_trigonometry, 'unique', 'dfs')


def retrieve_ops(expr):
    """Shorthand to retrieve the arithmetic operations within ``expr``."""
    return search(expr, q_op, 'all', 'dfs')

from devito.dse.queries import q_indexed, q_terminal, q_leaf, q_op, q_trigonometry

__all__ = ['retrieve_indexed', 'retrieve_terminals', 'retrieve_ops',
           'retrieve_trigonometry']


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

    def __init__(self, query, mode):
        """
        Search objects in an expression. This is much quicker than the more
        general SymPy's find.

        :param query: Any query from the ``queries`` module.
        :param mode: Either 'unique' or 'all' (catch all instances).
        """
        self.query = query
        self.collection = self.modes[mode]

    def _next(self, expr):
        return [] if q_leaf(expr) else expr.args

    def dfs(self, expr):
        """
        Perform a DFS search.

        :param expr: The searched expression
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

        :param expr: The searched expression
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

        :param expr: The searched expression
        """
        found = self.collection()
        if self.query(expr):
            found.update(self.collection.wrap(expr))
            return found
        for a in self._next(expr):
            found.update(self.bfs_first_hit(a))
        return found


def search(expr, query, mode='unique', visit='dfs'):
    """
    Interface to Search.
    """

    assert mode in Search.modes, "Unknown mode"
    assert visit in ['dfs', 'bfs', 'bfs_first_hit'], "Unknown visit type"

    searcher = Search(query, mode)
    if visit == 'dfs':
        return searcher.dfs(expr)
    elif visit == 'bfs':
        return searcher.bfs(expr)
    else:
        return searcher.bfs_first_hit(expr)


# Shorthands


def retrieve_indexed(expr, mode='unique'):
    """
    Shorthand to retrieve :class:`Indexed` objects in ``expr``.
    """
    return search(expr, q_indexed, mode, 'dfs')


def retrieve_terminals(expr, mode='unique'):
    """
    Shorthand to retrieve :class:`Indexed` and :class:`Symbol` objects in ``expr``.
    """
    return search(expr, q_terminal, mode, 'dfs')


def retrieve_trigonometry(expr):
    """
    Shorthand to retrieve trigonometric function objects in ``expr``.
    """
    return search(expr, q_trigonometry, 'unique', 'dfs')


def retrieve_ops(expr):
    """
    Shorthand to retrieve arithmetic operations rooted in ``expr``.
    """
    return search(expr, q_op, 'all', 'dfs')

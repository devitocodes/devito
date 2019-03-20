from collections import Callable, Iterable, OrderedDict, Mapping, deque
from functools import reduce

import numpy as np
from multidict import MultiDict

from devito.tools.utils import as_tuple, filter_ordered
from devito.tools.algorithms import toposort

__all__ = ['Bunch', 'EnrichedTuple', 'ReducerMap', 'DefaultOrderedDict',
           'PartialOrderTuple', 'DAG']


class Bunch(object):
    """
    Bind together an arbitrary number of generic items. This is a mutable
    alternative to a ``namedtuple``.

    From: ::

        http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of\
        -a-bunch-of-named/?in=user-97991
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class EnrichedTuple(tuple):
    """A tuple with an arbitrary number of additional attributes."""
    def __new__(cls, *items, getters=None, **kwargs):
        obj = super(EnrichedTuple, cls).__new__(cls, items)
        obj.__dict__.update(kwargs)
        obj._getters = dict(zip(getters or [], items))
        return obj

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return super(EnrichedTuple, self).__getitem__(key)
        else:
            return self._getters[key]


class ReducerMap(MultiDict):
    """
    Specialised :class:`MultiDict` object that maps a single key to a
    list of potential values and provides a reduction method for
    retrieval.
    """

    def update(self, values):
        """
        Update internal mapping with standard dictionary semantics.
        """
        if isinstance(values, Mapping):
            self.extend(values)
        elif isinstance(values, Iterable) and not isinstance(values, str):
            for v in values:
                self.extend(v)
        else:
            self.extend(values)

    def unique(self, key):
        """
        Returns a unique value for a given key, if such a value
        exists, and raises a ``ValueError`` if it does not.

        Parameters
        ----------
        key : str
            Key for which to retrieve a unique value.
        """
        candidates = self.getall(key)

        def compare_to_first(v):
            first = candidates[0]
            if isinstance(first, np.ndarray) or isinstance(v, np.ndarray):
                return (first == v).all()
            else:
                return first == v

        if len(candidates) == 1:
            return candidates[0]
        elif all(map(compare_to_first, candidates)):
            return candidates[0]
        else:
            raise ValueError("Unable to find unique value for key %s, candidates: %s"
                             % (key, candidates))

    def reduce(self, key, op=None):
        """
        Returns a reduction of all candidate values for a given key.

        Parameters
        ----------
        key : str
            Key for which to retrieve candidate values.
        op : callable, optional
            Operator for reduction among candidate values.  If not provided, a
            unique value will be returned.

        Raises
        ------
        ValueError
            If op is None and no unique value exists.
        """
        if op is None:
            # Return a unique value if it exists
            return self.unique(key)
        else:
            return reduce(op, self.getall(key))

    def reduce_all(self):
        """Returns a dictionary with reduced/unique values for all keys."""
        return {k: self.reduce(key=k) for k in self}


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)


class PartialOrderTuple(tuple):
    """
    A tuple whose elements are ordered according to a set of relations.

    Parameters
    ----------
    items : object or iterable of objects
        The elements of the tuple.
    relations : iterable of tuples, optional
        One or more binary relations between elements in ``items``. If not
        provided, then ``items`` is interpreted as a totally ordered sequence.
        If provided, then a (partial) ordering is computed and all elements in
        ``items`` for which a relation is not provided are appended.
    """
    def __new__(cls, items=None, relations=None):
        items = as_tuple(items)
        if relations:
            items = cls.reorder(items, relations)
        obj = super(PartialOrderTuple, cls).__new__(cls, items)
        obj._relations = set(tuple(i) for i in as_tuple(relations))
        return obj

    @classmethod
    def reorder(cls, items, relations):
        return filter_ordered(toposort(relations) + list(items))

    def __eq__(self, other):
        return super(PartialOrderTuple, self).__eq__(other) and\
            self.relations == other.relations

    def __hash__(self):
        return hash(*([i for i in self] + list(self.relations)))

    @property
    def relations(self):
        return self._relations

    def generate_ordering(self):
        raise NotImplementedError


class DAG(object):
    """
    A trivial implementation of a directed acyclic graph (DAG).

    Originally extracted from:

        https://github.com/thieman/py-dag/
    """

    def __init__(self, nodes=None, edges=None):
        self.graph = OrderedDict()
        for node in as_tuple(nodes):
            self.add_node(node)
        for ind_node, dep_node in as_tuple(edges):
            self.add_edge(ind_node, dep_node)

    def __contains__(self, key):
        return key in self.graph

    @property
    def nodes(self):
        return tuple(self.graph)

    @property
    def edges(self):
        ret = []
        for k, v in self.graph.items():
            ret.extend([(k, i) for i in v])
        return tuple(ret)

    def add_node(self, node_name, ignore_existing=False):
        """Add a node if it does not exist yet, or error out."""
        if node_name in self.graph:
            if ignore_existing is True:
                return
            raise KeyError('node %s already exists' % node_name)
        self.graph[node_name] = set()

    def delete_node(self, node_name):
        """Delete a node and all edges referencing it."""
        if node_name not in self.graph:
            raise KeyError('node %s does not exist' % node_name)
        self.graph.pop(node_name)
        for node, edges in self.graph.items():
            if node_name in edges:
                edges.remove(node_name)

    def add_edge(self, ind_node, dep_node, force_add=False):
        """Add an edge (dependency) between the specified nodes."""
        if force_add is True:
            self.add_node(ind_node, True)
            self.add_node(dep_node, True)
        if ind_node not in self.graph or dep_node not in self.graph:
            raise KeyError('one or more nodes do not exist in graph')
        self.graph[ind_node].add(dep_node)

    def delete_edge(self, ind_node, dep_node):
        """Delete an edge from the graph."""
        if dep_node not in self.graph.get(ind_node, []):
            raise KeyError('this edge does not exist in graph')
        self.graph[ind_node].remove(dep_node)

    def predecessors(self, node):
        """Return a list of all predecessors of the given node."""
        return [key for key in self.graph if node in self.graph[key]]

    def downstream(self, node):
        """Return a list of all nodes this node has edges towards."""
        if node not in self.graph:
            raise KeyError('node %s is not in graph' % node)
        return list(self.graph[node])

    def all_downstreams(self, node):
        """
        Return a list of all nodes ultimately downstream of the given node
        in the dependency graph, in topological order.
        """
        nodes = [node]
        nodes_seen = set()
        i = 0
        while i < len(nodes):
            downstreams = self.downstream(nodes[i])
            for downstream_node in downstreams:
                if downstream_node not in nodes_seen:
                    nodes_seen.add(downstream_node)
                    nodes.append(downstream_node)
            i += 1
        return list(filter(lambda node: node in nodes_seen,
                           self.topological_sort()))

    @property
    def size(self):
        return len(self.graph)

    def topological_sort(self):
        """
        Return a topological ordering of the DAG.

        Raises
        ------
        ValueError
            If it is not possible to compute a topological ordering, as the graph
            is invalid.
        """
        in_degree = {}
        for u in self.graph:
            in_degree[u] = 0

        for u in self.graph:
            for v in self.graph[u]:
                in_degree[v] += 1

        queue = deque()
        for u in in_degree:
            if in_degree[u] == 0:
                queue.appendleft(u)

        l = []
        while queue:
            u = queue.pop()
            l.append(u)
            for v in self.graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.appendleft(v)

        if len(l) == len(self.graph):
            return l
        else:
            raise ValueError('graph is not acyclic')

from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, MutableSet, Mapping, Set
from functools import reduce

import numpy as np
from multidict import MultiDict

from devito.tools import Pickable
from devito.tools.utils import as_tuple, filter_ordered
from devito.tools.algorithms import toposort

__all__ = ['Bunch', 'EnrichedTuple', 'ReducerMap', 'DefaultOrderedDict',
           'OrderedSet', 'PartialOrderTuple', 'DAG', 'frozendict',
           'UnboundedMultiTuple']


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


class EnrichedTuple(tuple, Pickable):

    """
    A tuple with an arbitrary number of additional attributes.
    """

    def __new__(cls, *items, getters=None, **kwargs):
        obj = super(EnrichedTuple, cls).__new__(cls, items)
        obj.__dict__.update(kwargs)
        obj._getters = OrderedDict(zip(getters or [], items))
        return obj

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, slice):
            items = super().__getitem__(key)
            if key.step is not None:
                return items
            # Reconstruct as an EnrichedTuple
            start = key.start or 0
            stop = key.stop if key.stop is not None else len(self)
            kwargs = dict(self.__dict__)
            kwargs['getters'] = list(self._getters)[start:stop]
            return EnrichedTuple(*items, **kwargs)
        else:
            return self.__getitem_hook__(key)

    def __getitem_hook__(self, key):
        return self._getters[key]

    def __getnewargs_ex__(self):
        # Bypass default reconstruction logic since this class spawns
        # objects with varying number of attributes
        return (tuple(self), dict(self.__dict__))


class ReducerMap(MultiDict):

    """
    Specialised MultiDict object that maps a single key to a
    list of potential values and provides a reduction method for
    retrieval.
    """

    @classmethod
    def fromdicts(cls, *dicts):
        ret = ReducerMap()
        for i in dicts:
            if not isinstance(i, Mapping):
                raise ValueError("Expected Mapping, got `%s`" % type(i))
            ret.update(i)
        return ret

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
            elif isinstance(v, Set):
                if isinstance(first, Set):
                    return not v.isdisjoint(first)
                else:
                    return first in v
            elif isinstance(first, Set):
                return v in first
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

    def reduce_inplace(self):
        """
        Like `reduce_all`, but it modifies self inplace, rather than
        returning the result as a new dict.
        """
        for k, v in self.reduce_all().items():
            self[k] = v


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


class OrderedSet(OrderedDict, MutableSet):

    """
    A simple implementation of an ordered set.

    Notes
    -----
    Readapted from:

        https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        for e in args:
            self.add(e)

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                self.add(e)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)
    union = property(lambda self: self.__or__)


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
    An implementation of a directed acyclic graph (DAG).

    Notes
    -----
    Originally extracted from:

        https://github.com/thieman/py-dag/
    """

    def __init__(self, nodes=None, edges=None):
        self.graph = OrderedDict()
        self.labels = DefaultOrderedDict(dict)
        for node in as_tuple(nodes):
            self.add_node(node)
        for i in as_tuple(edges):
            try:
                ind_node, dep_node = i
            except ValueError:
                ind_node, dep_node, label = i
                self.labels[ind_node][dep_node] = label
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

    @property
    def size(self):
        return len(self.graph)

    def add_node(self, node_name, ignore_existing=False):
        """Add a node if it does not exist yet, or error out."""
        if node_name in self.graph:
            if ignore_existing is True:
                return
            raise KeyError('node %s already exists' % node_name)
        self.graph[node_name] = OrderedSet()

    def delete_node(self, node_name):
        """Delete a node and all edges referencing it."""
        if node_name not in self.graph:
            raise KeyError('node %s does not exist' % node_name)
        self.graph.pop(node_name)
        for node, edges in self.graph.items():
            if node_name in edges:
                edges.remove(node_name)

    def add_edge(self, ind_node, dep_node, force_add=False, label=None):
        """Add an edge (dependency) between the specified nodes."""
        if force_add is True:
            self.add_node(ind_node, True)
            self.add_node(dep_node, True)
        if ind_node not in self.graph or dep_node not in self.graph:
            raise KeyError('one or more nodes do not exist in graph')
        self.graph[ind_node].add(dep_node)
        if label is not None:
            self.labels[ind_node][dep_node] = label

    def delete_edge(self, ind_node, dep_node):
        """Delete an edge from the graph."""
        if dep_node not in self.graph.get(ind_node, []):
            raise KeyError('this edge does not exist in graph')
        self.graph[ind_node].remove(dep_node)
        try:
            del self.labels[ind_node][dep_node]
        except KeyError:
            pass

    def get_label(self, ind_node, dep_node, default=None):
        try:
            return self.labels[ind_node][dep_node]
        except KeyError:
            return default

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
        nodes_seen = OrderedSet()
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

    def topological_sort(self, choose_element=None):
        """
        Return a topological ordering of the DAG.

        Parameters
        ----------
        choose_element : callable, optional
            A callback to pick an element out of the current candidates (i.e.,
            all un-scheduled nodes with no incoming edges). The callback takes
            in input an iterable of schedulable nodes as well as the list of
            already scheduled nodes; it must remove and return the selected node.

        Raises
        ------
        ValueError
            If it is not possible to compute a topological ordering, as the graph
            is invalid.
        """
        if choose_element is None:
            choose_element = lambda q, l: q.pop()

        in_degree = OrderedDict()  # OrderedDict, not dict, for determinism
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
            u = choose_element(queue, l)
            l.append(u)
            for v in self.graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.appendleft(v)

        if len(l) == len(self.graph):
            return l
        else:
            raise ValueError('graph is not acyclic')

    def connected_components(self, enumerated=False):
        """
        Find all connected sub-graphs and return them as a list.
        """
        groups = []

        for n0 in self.graph:
            found = {n0} | set(self.all_downstreams(n0))
            for g in groups:
                if g.intersection(found):
                    g.update(found)
                    break
            else:
                groups.append(found)

        if enumerated:
            mapper = OrderedDict()
            for n, g in enumerate(groups):
                mapper.update({i: n for i in g})
            return mapper
        else:
            return tuple(groups)


class frozendict(Mapping):
    """
    An immutable wrapper around dictionaries that implements the complete
    :py:class:`collections.Mapping` interface. It can be used as a drop-in
    replacement for dictionaries where immutability is desired.

    Extracted from the now decrepit project:

        https://github.com/slezica/python-frozendict
    """

    dict_cls = dict

    def __init__(self, *args, **kwargs):
        self._dict = self.dict_cls(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self._dict)

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self._dict.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash


class UnboundedMultiTuple(object):

    """
    An UnboundedMultiTuple is an ordered collection of tuples that can be
    infinitely iterated over.

    Examples
    --------
    >>> ub = UnboundedMultiTuple([1, 2], [3, 4])
    >>> ub
    UnboundedMultiTuple((1, 2), (3, 4))
    >>> ub.iter()
    >>> ub
    UnboundedMultiTuple(*(1, 2), (3, 4))
    >>> ub.next()
    1
    >>> ub.next()
    2
    >>> ub.iter()
    >>> ub.iter()  # No effect, tip has reached the last tuple
    >>> ub.iter()  # No effect, tip has reached the last tuple
    >>> ub
    UnboundedMultiTuple((1, 2), *(3, 4))
    >>> ub.next()
    3
    >>> ub.next()
    4
    >>> ub.iter()  # Reloads the last iterator
    >>> ub.next()
    3
    """

    def __init__(self, *items):
        # Normalize input
        nitems = []
        for i in as_tuple(items):
            if isinstance(i, Iterable):
                nitems.append(tuple(i))
            else:
                raise ValueError("Expected sequence, got %s" % type(i))

        self.items = tuple(nitems)
        self.tip = -1
        self.curiter = None

    def __repr__(self):
        items = [str(i) for i in self.items]
        if self.curiter is not None:
            items[self.tip] = "*%s" % items[self.tip]
        return "%s(%s)" % (self.__class__.__name__, ", ".join(items))

    def iter(self):
        if not self.items:
            raise ValueError("No tuples available")
        self.tip = min(self.tip + 1, max(len(self.items) - 1, 0))
        self.curiter = iter(self.items[self.tip])

    def next(self):
        if self.curiter is None:
            raise StopIteration
        return next(self.curiter)

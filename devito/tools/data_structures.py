from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, MutableSet, Mapping, Set
from functools import reduce, cached_property
import json

import numpy as np
from multidict import MultiDict

from devito.tools import Pickable
from devito.tools.utils import as_tuple, filter_ordered, humanbytes
from devito.tools.algorithms import toposort

__all__ = ['Bunch', 'EnrichedTuple', 'ReducerMap', 'DefaultOrderedDict',
           'OrderedSet', 'Ordering', 'DAG', 'frozendict',
           'UnboundTuple', 'UnboundedMultiTuple', 'MemoryEstimate']


class Bunch:

    """
    Bind together an arbitrary number of generic items. This is a mutable
    alternative to a ``namedtuple``.

    From: ::

        http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of\
        -a-bunch-of-named/?in=user-97991
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return "Bunch(%s)" % ", ".join(["%s=%s" % i for i in self.__dict__.items()])

    def __iter__(self):
        for i in self.__dict__.values():
            yield i


class EnrichedTuple(tuple, Pickable):

    """
    A tuple with an arbitrary number of additional attributes.
    """

    __rargs__ = ('*items',)
    __rkwargs__ = ('getters',)

    def __new__(cls, *items, getters=None, **kwargs):
        obj = super().__new__(cls, items)
        obj.__dict__.update(kwargs)
        # Convert to list if we're getting an OrderedDict from rebuild
        obj.getters = OrderedDict(zip(list(getters or []), items))
        return obj

    def _rebuild(self, *args, **kwargs):
        # Need to explicitly apply any additional attributes
        _kwargs = dict(self.__dict__)
        _kwargs.update(**kwargs)

        return super()._rebuild(*args, **_kwargs)

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
            kwargs['getters'] = list(self.getters)[start:stop]
            return EnrichedTuple(*items, **kwargs)
        else:
            return self.__getitem_hook__(key)

    def __getitem_hook__(self, key):
        return self.getters[key]

    def __getnewargs_ex__(self):
        # Bypass default reconstruction logic since this class spawns
        # objects with varying number of attributes
        sdict = {k: v for k, v in self.__dict__.items() if k not in self.getters}
        return tuple(self), sdict

    def get(self, key, val=None):
        return self.getters.get(key, val)

    @property
    def items(self) -> tuple:
        # Needed for rargs
        return tuple(self)


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
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return None

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
            elif isinstance(v, range):
                if isinstance(first, range):
                    return first.stop > v.start or v.stop > first.start
                else:
                    return first >= v.start and first < v.stop
            elif isinstance(first, range):
                return v >= first.start and v < first.stop
            else:
                return first == v

        if len(candidates) == 1:
            return candidates[0]
        elif all(map(compare_to_first, candidates)):
            # return first non-range
            for c in candidates:
                if not isinstance(c, range):
                    return c
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
        return type(self), args, None, None, self()

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

    def union(self, *args):
        ret = OrderedSet(*self)
        ret.update(*args)
        return ret

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


class Ordering(tuple):

    """
    A tuple whose elements are ordered according to a set of relations.

    Parameters
    ----------
    items : object or iterable of objects
        The elements of the tuple.
    relations : iterable of tuples, optional
        One or more n-ary relations between the elements in `items`. If not
        provided, then `items` is interpreted as a totally ordered sequence.
        If provided, then a (partial) ordering is computed and all elements in
        `items` for which a relation is not provided are appended.
    mode : str, optional
        If 'total' (default), the resulting object is interpreted as a totally
        ordered sequence; the object's relations are simplified away and no
        subsequent operation involving the Ordering will ever be able to alter
        the obtained sequence. If 'partial', the outcome is a partially ordered
        sequence; the relations as provided by the user are preserved, which
        leaves room for further reordering upon future operations. If 'unordered',
        the `relations` are ignored and the resulting object degenerates to an
        unordered collection.
    """
    def __new__(cls, items=None, relations=None, mode='total'):
        assert mode in ('total', 'partial', 'unordered')

        items = as_tuple(items)
        if relations:
            items = cls.reorder(items, relations)

        obj = super().__new__(cls, items)

        obj._relations = frozenset(cls.simplify_relations(relations, items, mode))
        obj._mode = mode

        return obj

    @classmethod
    def reorder(cls, items, relations):
        return filter_ordered(toposort(relations) + list(items))

    @classmethod
    def simplify_relations(cls, relations, items, mode):
        if mode == 'total':
            return [tuple(items)]
        elif mode == 'partial':
            return [tuple(i) for i in as_tuple(relations)]
        else:
            return []

    def __eq__(self, other):
        return (super().__eq__(other) and
                self.relations == other.relations and
                self.mode == other.mode)

    def __hash__(self):
        return hash(*([i for i in self] + list(self.relations) + [self.mode]))

    @property
    def relations(self):
        return self._relations

    @property
    def mode(self):
        return self._mode


class DAG:

    """
    An implementation of a directed acyclic graph (DAG).

    Notes
    -----
    Originally extracted from:

        https://github.com/thieman/py-dag/
    """

    def __init__(self, nodes=None, edges=None, labels=None):
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

        for ind_node, i in (labels or {}).items():
            for dep_node, v in i.items():
                if dep_node in self.graph.get(ind_node, []):
                    self.labels[ind_node][dep_node] = v

    def __contains__(self, key):
        return key in self.graph

    @property
    def nodes(self):
        return tuple(self.graph)

    @property
    def roots(self):
        return tuple(n for n in self.nodes if not self.predecessors(n))

    @property
    def edges(self):
        ret = []
        for k, v in self.graph.items():
            ret.extend([(k, i) for i in v])
        return tuple(ret)

    @property
    def size(self):
        return len(self.graph)

    def clone(self):
        return self.__class__(self.nodes, self.edges, self.labels)

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

    def all_predecessors(self, node):
        """
        Return a list of all nodes ultimately predecessors of the given node
        in the dependency graph, in topological order.
        """
        found = set()

        def _all_predecessors(n):
            if n in found:
                return
            found.add(n)
            for predecessor in self.predecessors(n):
                _all_predecessors(predecessor)

        _all_predecessors(node)

        return found

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

    def find_paths(self, node):
        if node not in self.graph:
            raise KeyError('node %s is not in graph' % node)

        paths = []

        def dfs(node, path):
            path.append(node)

            if not self.graph[node]:
                paths.append(tuple(path))
            else:
                for child in self.graph[node]:
                    dfs(child, path)

            # Remove the node from the path to backtrack
            path.pop()

        dfs(node, [])

        return tuple(paths)


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


class MemoryEstimate(frozendict):
    """
    An immutable mapper for a memory estimate, providing the estimated memory
    consumption across host, device, and so forth.

    Properties
    ----------
    name: str
        The name of the Operator for which this estimate was generated
    human_readable: frozendict
        The mapper, albeit with human-readable memory usage (MB, GB, etc)
        rather than raw bytes.

    Methods
    -------
    to_json(path)
        Write the memory estimate to a JSON for scheduler ingestion.
    """

    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', 'memory_estimate')
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return self._name

    @cached_property
    def human_readable(self):
        """The memory estimate in human-readable format"""
        return frozendict({k: humanbytes(v) for k, v in self.items()})

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name}): {self.human_readable._dict}'

    def to_json(self, path):
        """
        Write the MemoryEstimate to JSON for ingestion by a scheduler.

        Arguments
        ---------
        path: str
            The path to which the memory estimate should be written.
        """
        summary = {'name': self.name, **self._dict}
        json_object = json.dumps(summary, indent=4)

        with open(path, "w") as outfile:
            outfile.write(json_object)


class UnboundTuple(tuple):
    """
    An UnboundedTuple is a tuple that can be
    infinitely iterated over.

    Examples
    --------
    >>> ub = UnboundTuple((1, 2),(3, 4))
    >>> ub
    UnboundTuple(UnboundTuple(1, 2), UnboundTuple(3, 4))
    >>> ub.next()
    UnboundTuple(1, 2)
    >>> ub.next()
    UnboundTuple(3, 4)
    >>> ub.next()
    UnboundTuple(3, 4)
    """

    def __new__(cls, *items, **kwargs):
        nitems = []
        for i in as_tuple(items):
            if isinstance(i, UnboundTuple):
                nitems.append(i)
            elif isinstance(i, Iterable):
                nitems.append(UnboundTuple(*i))
            else:
                nitems.append(i)

        obj = super().__new__(cls, tuple(nitems))
        obj.last = len(nitems)
        obj.current = 0

        return obj

    @property
    def prod(self):
        return np.prod(self)

    def reset(self):
        self.iter()
        return self

    def iter(self):
        self.current = 0

    def next(self):
        if not self:
            return None
        item = self[self.current]
        if self.current == self.last-1 or self.current == -1:
            self.current = self.last
        else:
            self.current += 1
        return item

    def __len__(self):
        return self.last

    def __repr__(self):
        sitems = [s.__repr__() for s in self]
        return "%s(%s)" % (self.__class__.__name__, ", ".join(sitems))

    def __getitem__(self, idx):
        if not self:
            return None
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or self.last
            if stop < 0:
                stop = self.last + stop
            step = idx.step or 1
            return UnboundTuple(*[self[i] for i in range(start, stop, step)])
        try:
            if idx >= self.last-1:
                return super().__getitem__(self.last-1)
            else:
                return super().__getitem__(idx)
        except TypeError:
            # Slice, ...
            return UnboundTuple(self[i] for i in idx)


class UnboundedMultiTuple(UnboundTuple):

    """
    An UnboundedMultiTuple is an ordered collection of tuples that can be
    infinitely iterated over.

    Examples
    --------
    >>> ub = UnboundedMultiTuple([1, 2], [3, 4])
    >>> ub
    UnboundedMultiTuple(UnboundTuple(1, 2), UnboundTuple(3, 4))
    >>> ub.iter()
    >>> ub
    UnboundedMultiTuple(UnboundTuple(1, 2), UnboundTuple(3, 4))
    >>> ub.next()
    1
    >>> ub.next()
    2
    >>> ub.iter()
    >>> ub.iter()  # No effect, tip has reached the last tuple
    >>> ub.iter()  # No effect, tip has reached the last tuple
    >>> ub
    UnboundedMultiTuple(UnboundTuple(1, 2), UnboundTuple(3, 4))
    >>> ub.next()
    3
    >>> ub.next()
    4
    >>> ub.iter()  # Reloads the last iterator
    >>> ub.next()
    3
    """

    def __new__(cls, *items, **kwargs):
        obj = super().__new__(cls, *items, **kwargs)
        # MultiTuple are un-initialized
        obj.current = None
        return obj

    def reset(self):
        self.current = None
        return self

    def curitem(self):
        if self.current is None:
            raise StopIteration
        if not self:
            return None
        return self[self.current]

    def nextitem(self):
        if not self:
            return None
        self.iter()
        return self.curitem()

    def index(self, item):
        return self.index(item)

    def iter(self):
        if self.current is None:
            self.current = 0
        else:
            self.current = min(self.current + 1, self.last - 1)
        self[self.current].reset()
        return

    def next(self):
        if not self:
            return None
        if self.current is None:
            raise StopIteration
        if self[self.current].current >= self[self.current].last:
            raise StopIteration
        return self[self.current].next()

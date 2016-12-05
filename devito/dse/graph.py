"""
In a DSE graph, a node is a temporary and an edge between two nodes n0 and n1
indicates that n1 reads n0. For example, given the excerpt: ::

    temp0 = a*b
    temp1 = temp0*c
    temp2 = temp0*d
    temp3 = temp1 + temp2
    ...

A section of the ``temporaries graph`` looks as follows: ::

    temp0 ---> temp1
      |          |
      |          |
      v          v
    temp2 ---> temp3

Temporaries graph are used for symbolic as well as loop-level transformations.
"""

from collections import OrderedDict, namedtuple

from sympy import (Eq, Indexed)

from devito.dse.inspection import terminals

__all__ = ['temporaries_graph']


class Temporary(Eq):

    """
    A special :class:`sympy.Eq` which keeps track of: ::

        - :class:`sympy.Eq` writing to ``self``
        - :class:`sympy.Eq` reading from ``self``

    A :class:`Temporary` is used as node in a temporaries graph.
    """

    def __new__(cls, lhs, rhs, **kwargs):
        reads = kwargs.pop('reads', [])
        readby = kwargs.pop('readby', [])
        scope = kwargs.pop('scope', 0)
        obj = super(Temporary, cls).__new__(cls, lhs, rhs, **kwargs)
        obj._reads = set(reads)
        obj._readby = set(readby)
        obj._scope = scope
        return obj

    @property
    def reads(self):
        return self._reads

    @property
    def readby(self):
        return self._readby

    @property
    def is_terminal(self):
        return len(self.readby) == 0

    @property
    def is_tensor(self):
        return isinstance(self.lhs, Indexed) and self.lhs.rank > 0

    @property
    def is_scalarizable(self):
        return not self.is_terminal and self.is_tensor

    @property
    def scope(self):
        return self._scope

    def __repr__(self):
        return "DSE(%s, reads=%s, readby=%s)" % (super(Temporary, self).__repr__(),
                                                 str(self.reads), str(self.readby))


class Trace(OrderedDict):

    """
    Assign a depth level to each temporary in a temporary graph.
    """

    def __init__(self, root, graph, *args, **kwargs):
        super(Trace, self).__init__(*args, **kwargs)
        self._root = root
        self._compute(graph)

    def _compute(self, graph):
        if self.root not in graph:
            return
        to_visit = [(graph[self.root], 0)]
        while to_visit:
            temporary, level = to_visit.pop(0)
            self.__setitem__(temporary.lhs, level)
            to_visit.extend([(graph[i], level + 1) for i in temporary.reads])

    @property
    def root(self):
        return self._root

    @property
    def length(self):
        return len(self)

    def intersect(self, other):
        return Trace(self.root, {}, [(k, v) for k, v in self.items() if k in other])

    def union(self, other):
        return Trace(self.root, {}, [(k, v) for k, v in self.items() + other.items()])


def temporaries_graph(temporaries, scope=0):
    """
    Create a temporaries graph given a list of :class:`sympy.Eq`.
    """

    mapper = OrderedDict()
    Node = namedtuple('Node', ['rhs', 'reads', 'readby'])

    for lhs, rhs in [i.args for i in temporaries]:
        reads = {i for i in terminals(rhs) if i in mapper}
        mapper[lhs] = Node(rhs, reads, set())
        for i in mapper[lhs].reads:
            assert i in mapper, "Illegal Flow"
            mapper[i].readby.add(lhs)

    nodes = [Temporary(k, v.rhs, reads=v.reads, readby=v.readby, scope=scope)
             for k, v in mapper.items()]

    return OrderedDict([(i.lhs, i) for i in nodes])

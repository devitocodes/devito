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

from sympy import Indexed

from devito.dimension import t
from devito.dse.extended_sympy import Eq
from devito.dse.inspection import (is_indirect, retrieve_indexed, terminals)
from devito.tools import flatten

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
    def identifier(self):
        return self.lhs.base.label.name if self.is_tensor else self.lhs.name

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
    def is_scalar(self):
        return not self.is_tensor

    @property
    def scope(self):
        return self._scope

    def construct(self, rule):
        """
        Create a new temporary starting from ``self`` replacing symbols in
        the equation as specified by the dictionary ``rule``.
        """
        reads = set(self.reads) - set(rule.keys()) | set(rule.values())
        rhs = self.rhs.xreplace(rule)
        return Temporary(self.lhs, rhs, reads=reads, readby=self.readby,
                         scope=self.scope)

    def __repr__(self):
        return "DSE(%s, reads=%s, readby=%s)" % (super(Temporary, self).__repr__(),
                                                 str(self.reads), str(self.readby))


class TemporariesGraph(OrderedDict):

    """
    A temporaries graph built on top of an OrderedDict.
    """

    @property
    def clusters(self):
        """
        Compute the clusters of the temporaries graph. A cluster is an ordered
        collection of scalar expressions that are necessary to compute a tensor value,
        plus the tensor value itself.

        Examples
        ========
        In the following list of expressions: ::

            temp1 = a*b
            temp2 = c
            temp3[k] = temp1 + temp2
            temp4[k] = temp2 + 5
            temp5 = d*e
            temp6 = f+g
            temp7[i] = temp5 + temp6 + temp4[k]

        There are three target expressions: temp3, temp4, temp7. There are therefore
        three clusters: ((temp1, temp2, temp3), (temp2, temp4), (temp5, temp6, temp7)).
        The first and second clusters share the expression temp2. Note that temp4 does
        not appear in the third cluster, as it is not a scalar.
        """
        targets = [v for v in self.values() if v.is_tensor]
        clusters = [self.trace(i.lhs) for i in targets]
        # Drop the non-scalar expressions in each cluster
        for cluster in clusters:
            for k, v in list(cluster.items()):
                if v.is_tensor and not v.is_terminal:
                    cluster.pop(k)
        return clusters

    @property
    def space_indices(self):
        for v in self.values():
            if v.is_terminal and not is_indirect(v.lhs):
                found = v.lhs.free_symbols - {t, v.lhs.base.label}
                return tuple(sorted(found, key=lambda i: v.lhs.indices.index(i)))
        return ()

    @property
    def space_shape(self):
        for v in self.values():
            if v.is_terminal and not is_indirect(v.lhs):
                found = v.lhs.free_symbols - {t, v.lhs.base.label}
                return tuple(i for i, j in zip(v.lhs.shape, v.lhs.indices) if j in found)
        return ()

    def trace(self, root):
        if root not in self:
            return []
        found = OrderedDict()
        queue = [(self[root], 0)]
        while queue:
            temporary, index = queue.pop(0)
            found.setdefault(index, []).append(temporary)
            queue.extend([(self[i], index + 1) for i in temporary.reads])
        # Sort output for determinism
        found = reversed(found.values())
        found = flatten(sorted(v, key=lambda i: i.identifier) for v in found)
        return temporaries_graph(found)

    def is_index(self, root):
        if root not in self:
            return False
        queue = [self[root]]
        while queue:
            temporary = queue.pop(0)
            if any(root in i.atoms() for i in retrieve_indexed(temporary)):
                # /root/ appears amongst the indices of /temporary/
                return True
            else:
                queue.extend([self[i] for i in temporary.readby])
        return False


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

    return TemporariesGraph([(i.lhs, i) for i in nodes])

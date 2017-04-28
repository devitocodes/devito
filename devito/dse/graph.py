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

from devito.dimension import x, y, z, t  # TODO: Generalize to arbitrary dimensions
from devito.dse.extended_sympy import Eq
from devito.dse.search import retrieve_indexed
from devito.dse.inspection import terminals
from devito.dse.queries import q_indirect
from devito.stencil import Stencil
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
        obj = super(Temporary, cls).__new__(cls, lhs, rhs, **kwargs)
        obj._reads = set(reads)
        obj._readby = set(readby)
        return obj

    @property
    def identifier(self):
        return self.lhs.base.label.name if self.is_tensor else self.lhs.name

    @property
    def function(self):
        return self.lhs.base.function

    @property
    def reads(self):
        return self._reads

    @property
    def readby(self):
        return self._readby

    @property
    def is_cyclic_readby(self):
        return self.lhs in self.readby

    @property
    def is_terminal(self):
        return (len(self.readby) == 0) or\
            (len(self.readby) == 1 and self.is_cyclic_readby)

    @property
    def is_tensor(self):
        return isinstance(self.lhs, Indexed) and self.lhs.rank > 0

    @property
    def is_scalar(self):
        return not self.is_tensor

    def construct(self, rule):
        """
        Create a new temporary starting from ``self`` replacing symbols in
        the equation as specified by the dictionary ``rule``.
        """
        reads = set(self.reads) - set(rule.keys()) | set(rule.values())
        rhs = self.rhs.xreplace(rule)
        return Temporary(self.lhs, rhs, reads=reads, readby=self.readby)

    def __repr__(self):
        return "DSE(%s, reads=%s, readby=%s)" % (super(Temporary, self).__repr__(),
                                                 str(self.reads), str(self.readby))


class TemporariesGraph(OrderedDict):

    """
    A temporaries graph built on top of an OrderedDict.
    """

    def clusters(self, domains):
        """
        Compute the clusters of the temporaries graph. See Cluster.__doc__ for
        more information about clusters.
        """
        targets = [v for v in self.values() if v.is_tensor]
        clusters = []
        for i in targets:
            # Determine what temporaries are needed to compute /i/
            trace = self.trace(i.lhs)
            # Compute the stencil of the cluster
            stencil = Stencil.union(*[domains.get(v.lhs, {}) for v in trace.values()])
            # Create and track the cluster
            clusters.append(Cluster(trace, stencil))
        return clusters

    @property
    def space_indices(self):
        seen = set()
        candidates = [x, y, z]
        terms = [k for k, v in self.items() if v.is_tensor and not q_indirect(k)]
        for term in terms:
            seen |= {i for i in term.base.function.indices if i in candidates}
        return tuple(sorted(seen, key=lambda i: candidates.index(i)))

    @property
    def space_shape(self):
        candidates = self.space_indices
        terms = [k for k, v in self.items() if v.is_tensor and not q_indirect(k)]
        for term in terms:
            indices = term.base.function.indices
            if set(candidates).issubset(set(indices)):
                return tuple(i for i, j in zip(term.shape, indices) if j in candidates)
        return ()

    def trace(self, root):
        if root not in self:
            return []
        found = OrderedDict()
        queue = [(self[root], 0)]
        while queue:
            item, index = queue.pop(0)
            found.setdefault(index, []).append(item)
            queue.extend([(self[i], index+1) for i in item.reads if self[i] != item])
        # Sort output for determinism
        found = reversed(found.values())
        found = flatten(sorted(v, key=lambda i: i.identifier) for v in found)
        return temporaries_graph(found)

    def time_invariant(self, expr=None):
        """
        Check if ``expr`` is time invariant. ``expr`` may be an expression ``e``
        explicitly tracked by the TemporariesGraph or even a generic subexpression
        of ``e``. If no ``expr`` is provided, then time invariance is checked
        on the entire TemporariesGraph.
        """
        if expr is None:
            return all(self.time_invariant(v) for v in self.values())

        if t in expr.free_symbols:
            return False
        to_visit = [expr.rhs] if expr.is_Equality else [expr]
        while to_visit:
            item = to_visit.pop()
            for i in retrieve_indexed(item):
                if t in i.free_symbols:
                    return False
            temporaries = [i for i in item.free_symbols if i in self]
            to_visit.extend([self[i].rhs for i in temporaries if self[i].rhs != item])
        return True

    def is_index(self, root):
        if root not in self:
            return False
        queue = [self[root]]
        while queue:
            item = queue.pop(0)
            if any(root in i.atoms() for i in retrieve_indexed(item)):
                # /root/ appears amongst the indices of /item/
                return True
            else:
                queue.extend([self[i] for i in item.readby if self[i] != item])
        return False


class Cluster(object):

    """
    A Cluster is an ordered collection of expressions that are necessary to
    compute a tensor, plus the tensor expression itself.

    A Cluster is associated with a "stencil", which tracks what neighboring points
    are required, along each dimension, to compute an entry in the tensor.

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

    @classmethod
    def merge(cls, clusters):
        """
        Given an ordered collection of :class:`Cluster` objects, return a
        (potentially) smaller sequence in which clusters with identical stencil
        have been merged.
        """
        mapper = OrderedDict()
        for c in clusters:
            mapper.setdefault(tuple(c.stencil.entries), []).append(c)

        processed = []
        for entries, clusters in mapper.items():
            temporaries = OrderedDict()
            for c in clusters:
                for k, v in c.trace.items():
                    if k not in temporaries:
                        temporaries[k] = v
            processed.append(SuperCluster(temporaries_graph(temporaries.values()),
                                          Stencil(entries)))

        return processed

    def __init__(self, trace, stencil):
        self._full_trace = trace
        self._stencil = stencil.frozen

        # Determine the output tensor
        output = [v for v in self.trace.values() if v.is_tensor]
        assert len(output) == 1
        self._output = output[0]

    def _view(self, drop=lambda v: False):
        cls = type(self._full_trace)
        return cls([(k, v) for k, v in self._full_trace.items() if not drop(v)])

    @property
    def output(self):
        return self._output

    @property
    def trace(self):
        """The ordered collection of expressions to compute the output tensor."""
        return self._view(lambda v: v.is_tensor and not v.is_terminal)

    @property
    def external(self):
        """Tensors from other clusters required to compute the output tensor."""
        return self._view(lambda v: v.is_scalar or v == self.output)

    @property
    def needs(self):
        handle = flatten(retrieve_indexed(v.rhs) for v in self._view().values())
        handle = {v.base.function for v in handle}
        return [v for v in handle if not v.is_SymbolicData]

    @property
    def stencil(self):
        return self._stencil


class SuperCluster(object):

    """
    A SuperCluster represents the result of merging a collection of cluster.
    """

    def __init__(self, trace, stencil):
        self.trace = trace
        self.stencil = stencil


def temporaries_graph(temporaries):
    """
    Create a dependency graph given a list of :class:`sympy.Eq`.
    """

    # Check input is legal and initialize the temporaries graph
    temporaries = [Temporary(*i.args) for i in temporaries]
    nodes = [i.lhs for i in temporaries]
    assert len(set(nodes)) == len(nodes)
    graph = TemporariesGraph(zip(nodes, temporaries))

    # Add edges (i.e., reads and readby info) to the graph
    mapper = OrderedDict()
    for i in nodes:
        mapper.setdefault(as_symbol(i), []).append(i)
    for k, v in graph.items():
        handle = terminals(v.rhs)
        v.reads.update(set(flatten([mapper.get(as_symbol(i), []) for i in handle])))
        for i in v.reads:
            graph[i].readby.add(k)

    return graph

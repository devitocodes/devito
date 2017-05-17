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

from collections import OrderedDict
from itertools import islice

from sympy import Indexed

from devito.dimension import x, y, z, t, time
from devito.dse.extended_sympy import Eq
from devito.dse.search import retrieve_indexed
from devito.dse.inspection import as_symbol, terminals
from devito.dse.queries import q_indirect
from devito.exceptions import DSEException
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
    def shape(self):
        return self.lhs.shape if self.is_tensor else ()

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

    @property
    def is_dead(self):
        return self.is_scalar and self.is_terminal and len(self.reads) == 1

    def construct(self, rule):
        """
        Create a new temporary starting from ``self`` replacing symbols in
        the equation as specified by the dictionary ``rule``.
        """
        reads = set(self.reads) - set(rule.keys()) | set(rule.values())
        rhs = self.rhs.xreplace(rule)
        return Temporary(self.lhs, rhs, reads=reads, readby=self.readby)

    def __repr__(self):
        reads = '[%s%s]' % (', '.join([str(i) for i in self.reads][:2]), '%s')
        reads = reads % ('' if len(self.reads) <= 2 else ', ...')
        readby = '[%s%s]' % (', '.join([str(i) for i in self.readby][:2]), '%s')
        readby = readby % ('' if len(self.readby) <= 2 else ', ...')
        return "Temp(key=%s, reads=%s, readby=%s)" % (self.lhs, reads, readby)


class TemporariesGraph(OrderedDict):

    """
    A temporaries graph represents an ordered sequence of operations.

    The operations may involve scalars and indexed objects (arrays). The indices
    of the indexed objects represent either "space" or "time" dimensions.
    """

    def __init__(self, *args, **kwargs):
        super(TemporariesGraph, self).__init__(*args, **kwargs)

        # TODO: The following need to be generalized to arbitrary dimensions,
        # not just x, y, z, t, time

        terms = [v for k, v in self.items() if v.is_tensor and not q_indirect(k)]

        # Determine the indices along the space dimensions
        candidates = [x, y, z]
        seen = set()
        for i in terms:
            seen |= {j for j in i.function.indices if j in candidates}
        self.space_indices = tuple(sorted(seen, key=lambda i: candidates.index(i)))

        # Determine the shape of the tensors in the spatial dimensions
        self.space_shape = ()
        for i in terms:
            if set(self.space_indices).issubset(set(i.function.indices)):
                self.space_shape = tuple(k for k, v in zip(i.shape, i.function.indices)
                                         if v in self.space_indices)
                break

        # Determine the indices along the time dimension
        self.time_indices = [t, time]

    def trace(self, key):
        """
        Return the sequence of operations required to compute the temporary ``key``.
        """
        if key not in self:
            return []

        # OrderedDicts, besides preserving the scheduling order, also prevent
        # scheduling the same temporary more than once
        found = OrderedDict()
        queue = OrderedDict([(key, self[key])])
        while queue:
            k, v = queue.popitem(last=False)
            reads = self.extract(k)
            if set(reads).issubset(set(found.values())):
                # All dependencies satisfied, schedulable
                found[k] = v
            else:
                # Tensors belong to other traces, so they can be scheduled straight away
                tensors = [i for i in reads if i.is_tensor]
                found = OrderedDict(list(found.items()) + [(i.lhs, i) for i in tensors])
                # Postpone the rest until all dependening temporaries got scheduled
                scalars = [i for i in reads if i.is_scalar]
                queue = OrderedDict([(i.lhs, i) for i in scalars] +
                                    [(k, v)] + list(queue.items()))
        return found.values()

    def time_invariant(self, expr=None):
        """
        Check if ``expr`` is time invariant. ``expr`` may be an expression ``e``
        explicitly tracked by the TemporariesGraph or even a generic subexpression
        of ``e``. If no ``expr`` is provided, then time invariance of the entire
        TemporariesGraph is assessed.
        """
        if expr is None:
            return all(self.time_invariant(v) for v in self.values())

        if any(i in expr.free_symbols for i in self.time_indices):
            return False
        queue = [expr.rhs] if expr.is_Equality else [expr]
        while queue:
            item = queue.pop()
            for i in retrieve_indexed(item):
                if any(j in i.free_symbols for j in self.time_indices):
                    return False
            temporaries = [i for i in item.free_symbols if i in self]
            queue.extend([self[i].rhs for i in temporaries if self[i].rhs != item])
        return True

    def is_index(self, key):
        """
        Return True if ``key`` is used as array index in an expression of the
        TemporariesGraph, False otherwise.
        """
        if key not in self:
            return False
        seen = set()
        queue = [self[key]]
        while queue:
            item = queue.pop(0)
            seen.add(item)
            if any(key in i.atoms() for i in retrieve_indexed(item)):
                # /key/ appears amongst the indices of /item/
                return True
            else:
                queue.extend([i for i in self.extract(item.lhs, readby=True)
                              if i not in seen])
        return False

    def extract(self, key, readby=False):
        """
        Return the list of nodes appearing in ``key.reads``, in program order
        (ie, based on the order in which the temporaries appear in ``self``). If
        ``readby is True``, then return instead the list of nodes appearing
        ``key.readby``, in program order.

        Examples
        ========
        Given the following sequence of operations: ::

            t1 = ...
            t0 = ...
            u[i, j] = ... v ...
            u[3, j] = ...
            v = t0 + t1 + u[z, k]
            t2 = ...

        Assuming ``key == v`` and ``readby is False`` (as by default), return
        the following list of :class:`Temporary` objects: ::

            [t1, t0, u[i, j], u[3, j]]

        If ``readby is True``, return: ::

            [v, t2]
        """
        if key not in self:
            return []
        match = self[key].reads if readby is False else self[key].readby
        found = []
        for k, v in self.items():
            if k in match:
                found.append(v)
        return found

    def __getitem__(self, key):
        if not isinstance(key, slice):
            return super(TemporariesGraph, self).__getitem__(key)
        offset = key.step or 0
        try:
            start = list(self.keys()).index(key.start) + offset
        except ValueError:
            start = 0
        try:
            stop = list(self.keys()).index(key.stop) + offset
        except ValueError:
            stop = None
        return TemporariesGraph(islice(list(self.items()), start, stop))


def temporaries_graph(temporaries):
    """
    Create a dependency graph given a list of :class:`sympy.Eq`.
    """

    # Check input is legal and initialize the temporaries graph
    temporaries = [Temporary(*i.args) for i in temporaries]
    nodes = [i.lhs for i in temporaries]
    if len(set(nodes)) != len(nodes):
        raise DSEException("Found redundant node in the TemporariesGraph.")
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

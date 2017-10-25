from collections import OrderedDict
from itertools import islice

from cached_property import cached_property
from sympy import Indexed

from devito.dimension import Dimension
from devito.symbolics import (Eq, as_symbol, retrieve_indexed, retrieve_terminals,
                              q_indirect, q_timedimension)
from devito.exceptions import DSEException
from devito.tools import flatten, filter_ordered

__all__ = ['temporaries_graph']


class Temporary(Eq):

    """
    A special :class:`sympy.Eq` which keeps track of: ::

        - :class:`sympy.Eq` writing to ``self``
        - :class:`sympy.Eq` reading from ``self``
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
    A temporaries graph represents an ordered sequence of operations. Operations,
    of type :class:`Temporary`, are the nodes of the graph. An edge from ``n0`` to
    ``n1`` indicates that ``n1`` reads from ``n0``. For example, the sequence: ::

        temp0 = a*b
        temp1 = temp0*c
        temp2 = temp0*d
        temp3 = temp1 + temp2

    is represented by the following TemporariesGraph: ::

        temp0 ---> temp1
          |          |
          |          |
          v          v
        temp2 ---> temp3

    The input and output edges of a node ``n`` are encoded in ``n.reads`` and
    ``n.readby``, respectively.

    Operations may involve scalars and indexed objects (arrays). The indices
    of the indexed objects represent either "space" or "time" dimensions.
    """

    def __init__(self, *args, **kwargs):
        super(TemporariesGraph, self).__init__(*args, **kwargs)

        terms = [v for k, v in self.items() if v.is_tensor and not q_indirect(k)]
        indices = filter_ordered(flatten([i.function.indices for i in terms]))

        # Determine indices along the space and time dimensions
        self.space_indices = tuple(i for i in indices if i.is_Space)
        self.time_indices = tuple(i for i in indices if i.is_Time)

    def trace(self, key, readby=False, strict=False):
        """
        Return the sequence of operations required to compute the temporary ``key``.
        If ``readby = True``, then return the sequence of operations that will
        depend on ``key``, instead. With ``strict = True``, drop ``key`` from the
        result.
        """
        if key not in self:
            return []

        # OrderedDicts, besides preserving the scheduling order, also prevent
        # scheduling the same temporary more than once
        found = OrderedDict()
        queue = OrderedDict([(key, self[key])])
        while queue:
            k, v = queue.popitem(last=False)
            reads = self.extract(k, readby=readby)
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
        if strict is True:
            found.pop(key)
        return tuple(found.values())

    def reschedule(self, context):
        """
        Starting from the temporaries in ``self``, return a new sequence of
        expressions that: ::

            * includes all expressions in ``context`` not appearing in ``self``, and
            * is ordered so that the ordering in ``context`` is honored.

        Examples
        ========
        Assume that: ::

            * ``self`` has five temporaries ``[t0, t1, t2, e1, e2]``,
            * ``t1`` depends on the temporary ``e1``, and ``t2`` depends on ``t1``
            * ``context = [e1, e2]``

        Then the following sequence is returned ``[t0, e1, t1, t2, e2]``.

        If, instead, we had had everything like before except: ::

            * ``context = [t1, e1, e2]``

        Then the following sequence is returned ``[t0, t1, t2, e1, e2]``.
        That is, in the latter example the original ordering dictated by ``context``
        was honored.
        """
        processed = [i.lhs for i in context]
        queue = [i for i in self if i not in processed]
        while queue:
            k = queue.pop(0)
            handle = self[k].readby
            if all(i in processed for i in handle):
                index = min(processed.index(i) for i in handle)
                processed.insert(index, k)
            else:
                # Note: push at the back
                queue.append(k)

        processed = [self[i] for i in processed]

        return processed

    def time_invariant(self, expr=None):
        """
        Check if ``expr`` is time invariant. ``expr`` may be an expression ``e``
        explicitly tracked by the TemporariesGraph or even a generic subexpression
        of ``e``. If no ``expr`` is provided, then time invariance of the entire
        TemporariesGraph is assessed.
        """
        if expr is None:
            return all(self.time_invariant(v) for v in self.values())

        if any(q_timedimension(i) for i in expr.free_symbols):
            return False

        queue = [expr.rhs] if expr.is_Equality else [expr]
        while queue:
            item = queue.pop()
            temporaries = []
            for i in retrieve_terminals(item):
                if any(isinstance(j, Dimension) and j.is_Time for j in i.free_symbols):
                    # Definitely not time-invariant
                    return False
                if i in self:
                    # Go on with the search
                    temporaries.append(i)
                elif isinstance(i, Dimension):
                    # Go on with the search, as /i/ is not a time dimension
                    continue
                elif not i.base.function.is_SymbolicFunction:
                    # It didn't come from the outside and it's not in self, so
                    # cannot determine if time-invariant; assume time-varying
                    return False
            queue.extend([self[i].rhs for i in temporaries if self[i].rhs != item])
        return True

    def is_index(self, key):
        """
        Return True if ``key`` is used as array index in an expression of the
        TemporariesGraph, False otherwise.
        """
        if key not in self:
            return False
        match = key.base.label if self[key].is_tensor else key
        for i in self.extract(key, readby=True):
            for e in retrieve_indexed(i):
                if any(match in idx.free_symbols for idx in e.indices):
                    return True
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

    @cached_property
    def unknown(self):
        """
        Return all symbols appearing in self for which a temporary is not available.
        """
        known = {v.function for v in self.values()}
        reads = set([i.base.function for i in
                     flatten(retrieve_terminals(v.rhs) for v in self.values())])
        return reads - known

    @cached_property
    def tensors(self):
        """
        Return all occurrences of the tensors in ``self`` keyed by function.
        """
        mapper = {}
        for v in self.values():
            handle = retrieve_indexed(v)
            for i in handle:
                found = mapper.setdefault(i.base.function, [])
                if i not in found:
                    # Not using sets to preserve order
                    found.append(i)
        return mapper


def temporaries_graph(temporaries):
    """
    Create a :class:`TemporariesGraph` given a list of :class:`sympy.Eq`.
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
        # Scalars
        handle = retrieve_terminals(v.rhs)

        # Tensors (does not inspect indirections such as A[B[i]])
        for i in list(handle):
            if i.is_Indexed:
                for idx in i.indices:
                    handle |= retrieve_terminals(idx)

        # Derive actual reads
        reads = set(flatten([mapper.get(as_symbol(i), []) for i in handle]))

        # Propagate information
        v.reads.update(reads)
        for i in v.reads:
            graph[i].readby.add(k)

    return graph

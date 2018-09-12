from collections import OrderedDict
from itertools import islice

from cached_property import cached_property

from devito.dimension import Dimension
from devito.ir.equations import ClusterizedEq
from devito.symbolics import (as_symbol, retrieve_indexed, retrieve_terminals,
                              convert_to_SSA, q_indirect, q_timedimension)
from devito.tools import DefaultOrderedDict, flatten, filter_ordered

__all__ = ['FlowGraph']


class Node(ClusterizedEq):

    """
    A special :class:`ClusterizedEq` which keeps track of: ::

        - :class:`sympy.Eq` writing to ``self``
        - :class:`sympy.Eq` reading from ``self``
    """

    _state = ClusterizedEq._state + ('reads', 'readby')

    @property
    def function(self):
        return self.lhs.function

    @property
    def reads(self):
        return self._reads

    @property
    def readby(self):
        return self._readby

    @property
    def is_unbound_temporary(self):
        return self.function.is_Array and not self.reads and not self.readby

    def __repr__(self):
        reads = '[%s%s]' % (', '.join([str(i) for i in self.reads][:2]), '%s')
        reads = reads % ('' if len(self.reads) <= 2 else ', ...')
        readby = '[%s%s]' % (', '.join([str(i) for i in self.readby][:2]), '%s')
        readby = readby % ('' if len(self.readby) <= 2 else ', ...')
        return "Node(key=%s, reads=%s, readby=%s)" % (self.lhs, reads, readby)


class FlowGraph(OrderedDict):

    """
    A FlowGraph represents an ordered sequence of operations. Operations,
    of type :class:`Node`, are the nodes of the graph. An edge from ``n0`` to
    ``n1`` indicates that ``n1`` reads from ``n0``. For example, the sequence: ::

        temp0 = a*b
        temp1 = temp0*c
        temp2 = temp0*d
        temp3 = temp1 + temp2

    is represented by the following FlowGraph: ::

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

    def __init__(self, exprs, **kwargs):
        # Always convert to SSA
        exprs = convert_to_SSA(exprs)
        mapper = OrderedDict([(i.lhs, i) for i in exprs])
        assert len(set(mapper)) == len(exprs), "not SSA Cluster?"

        # Construct the Nodes, tracking reads and readby
        tensor_map = DefaultOrderedDict(list)
        for i in mapper:
            tensor_map[as_symbol(i)].append(i)
        reads = DefaultOrderedDict(set)
        readby = DefaultOrderedDict(set)
        for k, v in mapper.items():
            handle = retrieve_terminals(v.rhs)
            for i in list(handle):
                if i.is_Indexed:
                    for idx in i.indices:
                        handle |= retrieve_terminals(idx)
            reads[k].update(set(flatten([tensor_map.get(as_symbol(i), [])
                                         for i in handle])))
            for i in reads[k]:
                readby[i].add(k)

        # Make sure read-after-writes are honored for scalar nodes
        processed = [i for i in mapper if i.is_Indexed]
        queue = [i for i in mapper if i not in processed]
        while queue:
            k = queue.pop(0)
            if not readby[k]:
                processed.insert(0, k)
            elif all(i in processed for i in readby[k]):
                index = min(processed.index(i) for i in readby[k])
                processed.insert(index, k)
            else:
                queue.append(k)

        # Build up the FlowGraph
        nodes = [(i, Node(mapper[i], reads=reads[i], readby=readby[i]))
                 for i in processed]
        super(FlowGraph, self).__init__(nodes, **kwargs)

        # Determine indices along the space and time dimensions
        terms = [v for k, v in self.items() if v.is_Tensor and not q_indirect(k)]
        indices = filter_ordered(flatten([i.function.indices for i in terms]))
        self.space_indices = tuple(i for i in indices if i.is_Space)
        self.time_indices = tuple(i for i in indices if i.is_Time)

    def trace(self, key, readby=False, strict=False):
        """
        Return the sequence of operations required to compute the node ``key``.
        If ``readby = True``, then return the sequence of operations that will
        depend on ``key``, instead. With ``strict = True``, drop ``key`` from the
        result.
        """
        if key not in self:
            return []

        # OrderedDicts, besides preserving the scheduling order, also prevent
        # scheduling the same node more than once
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
                tensors = [i for i in reads if i.is_Tensor]
                found = OrderedDict(list(found.items()) + [(i.lhs, i) for i in tensors])
                # Postpone the rest until all dependening nodes got scheduled
                scalars = [i for i in reads if i.is_Scalar]
                queue = OrderedDict([(i.lhs, i) for i in scalars] +
                                    [(k, v)] + list(queue.items()))
        if strict is True:
            found.pop(key)
        return tuple(found.values())

    def time_invariant(self, expr=None):
        """
        Check if ``expr`` is time invariant. ``expr`` may be an expression ``e``
        explicitly tracked by the FlowGraph or even a generic subexpression
        of ``e``. If no ``expr`` is provided, then time invariance of the entire
        FlowGraph is assessed.
        """
        if expr is None:
            return all(self.time_invariant(v) for v in self.values())

        if any(q_timedimension(i) for i in expr.free_symbols):
            return False

        queue = [expr.rhs if expr.is_Equality else expr]
        seen = set()
        while queue:
            item = queue.pop()
            nodes = set()
            for i in retrieve_terminals(item):
                if i in seen:
                    # Already inspected, nothing more can be inferred
                    continue
                elif any(isinstance(j, Dimension) and j.is_Time for j in i.free_symbols):
                    # Definitely not time-invariant
                    return False
                elif i in self:
                    # Go on with the search
                    nodes.add(i)
                elif isinstance(i, Dimension):
                    # Go on with the search, as /i/ is not a time dimension
                    pass
                elif not i.base.function.is_TensorFunction:
                    # It didn't come from the outside and it's not in self, so
                    # cannot determine if time-invariant; assume time-varying
                    return False
                seen.add(i)
            queue.extend([self[i].rhs for i in nodes])
        return True

    def is_index(self, key):
        """
        Return True if ``key`` is used as array index in an expression of the
        FlowGraph, False otherwise.
        """
        if key not in self:
            return False
        match = key.base.label if self[key].is_Tensor else key
        for i in self.extract(key, readby=True):
            for e in retrieve_indexed(i):
                if any(match in idx.free_symbols for idx in e.indices):
                    return True
        return False

    def extract(self, key, readby=False):
        """
        Return the list of nodes appearing in ``key.reads``, in program order
        (ie, based on the order in which the nodes appear in ``self``). If
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
        the following list of :class:`Node` objects: ::

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
            return super(FlowGraph, self).__getitem__(key)
        offset = key.step or 0
        try:
            start = list(self.keys()).index(key.start) + offset
        except ValueError:
            start = 0
        try:
            stop = list(self.keys()).index(key.stop) + offset
        except ValueError:
            stop = None
        return FlowGraph(islice(list(self.items()), start, stop))

    @cached_property
    def unknown(self):
        """
        Return all symbols appearing in self for which a node is not available.
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

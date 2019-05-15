from collections import OrderedDict
from itertools import islice

from cached_property import cached_property

from devito.ir.equations import ClusterizedEq
from devito.symbolics import (as_symbol, retrieve_terminals, q_timedimension)
from devito.tools import DefaultOrderedDict, flatten
from devito.types import Dimension, Symbol

__all__ = ['FlowGraph']


class Node(ClusterizedEq):

    """
    A special ClusterizedEq which keeps track of: ::

        - Equations writing to ``self``
        - Equations reading from ``self``
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
    A FlowGraph represents an ordered sequence of operations. The operations,
    objects of type Node, are the nodes of the graph. An edge from ``n0`` to
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

    Operations may involve scalars and indexed objects (arrays).
    """

    def __init__(self, exprs, **kwargs):
        # Always convert to SSA
        exprs = makeit_ssa(exprs)
        mapper = OrderedDict([(i.lhs, i) for i in exprs])
        assert len(set(mapper)) == len(exprs), "not SSA Cluster?"

        # Construct the Nodes, tracking reads and readby
        tensor_map = DefaultOrderedDict(list)
        for i in mapper:
            tensor_map[as_symbol(i)].append(i)
        reads = DefaultOrderedDict(set)
        readby = DefaultOrderedDict(set)
        for k, v in mapper.items():
            handle = retrieve_terminals(v.rhs, deep=True)
            reads[k].update(set(flatten([tensor_map.get(as_symbol(i), [])
                                         for i in handle])))
            for i in reads[k]:
                readby[i].add(k)

        # Make sure read-after-writes are honored for scalar nodes
        processed = [i for i in mapper if i.is_Indexed]
        queue = [i for i in mapper if i not in processed]
        while queue:
            k = queue.pop(0)
            if not readby[k] or k in readby[k]:
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
                elif not i.function.is_DiscreteFunction:
                    # It didn't come from the outside and it's not in self, so
                    # cannot determine if time-invariant; assume time-varying
                    return False
                seen.add(i)
            queue.extend([self[i].rhs for i in nodes])
        return True

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


def makeit_ssa(exprs):
    """Convert an iterable of Eqs into Static Single Assignment (SSA) form."""
    # Identify recurring LHSs
    seen = {}
    for i, e in enumerate(exprs):
        seen.setdefault(e.lhs, []).append(i)
    # Optimization: don't waste time reconstructing stuff if already in SSA form
    if all(len(i) == 1 for i in seen.values()):
        return exprs
    # SSA conversion
    c = 0
    mapper = {}
    processed = []
    for i, e in enumerate(exprs):
        where = seen[e.lhs]
        rhs = e.rhs.xreplace(mapper)
        if len(where) > 1:
            needssa = e.is_Scalar or where[-1] != i
            lhs = Symbol(name='ssa%d' % c, dtype=e.dtype) if needssa else e.lhs
            if e.is_Increment:
                # Turn AugmentedAssignment into Assignment
                processed.append(e.func(lhs, mapper[e.lhs] + rhs, is_Increment=False))
            else:
                processed.append(e.func(lhs, rhs))
            mapper[e.lhs] = lhs
            c += 1
        else:
            processed.append(e.func(e.lhs, rhs))
    return processed

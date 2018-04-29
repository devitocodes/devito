from collections import OrderedDict

from devito.ir.stree.tree import ScheduleTree, NodeIteration, NodeConditional, NodeExprs

__all__ = ['schedule']


def schedule(clusters):
    """
    Arrange an iterable of :class:`Cluster`s into a :class:`ScheduleTree`.
    """
    stree = ScheduleTree()

    mapper = OrderedDict()
    for c in clusters:
        pointers = list(mapper)

        # Find out if any of the existing nodes can be reused
        index = 0
        root = stree
        for it0, it1 in zip(c.itintervals, pointers):
            if it0 != it1 or it0.dim in c.atomics:
                break
            root = mapper[it0]
            index += 1
            if it0.dim in c.guards:
                break

        # The reused sub-trees might acquire some new sub-iterators
        for i in pointers[:index]:
            mapper[i].ispace.merge(c.ispace)
        # Later sub-trees, instead, will not be used anymore
        for i in pointers[index:]:
            mapper.pop(i)

        # Add in Iterations
        for i in c.itintervals[index:]:
            root = NodeIteration(c.ispace.project([i.dim]), root)
            mapper[i] = root

        # Add in Expressions
        NodeExprs(c.exprs, root)

        # Add in Conditionals
        for k, v in mapper.items():
            if k.dim in c.guards:
                node = NodeConditional(c.guards[k.dim])
                v.last.parent = node
                node.parent = v

    return stree

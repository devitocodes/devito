from collections import OrderedDict

from anytree import findall

from devito.ir.stree.tree import (ScheduleTree, NodeIteration, NodeConditional,
                                  NodeExprs, NodeSection)
from devito.tools import flatten

__all__ = ['schedule', 'section']


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
        NodeExprs(c.exprs, c.shape, c.ops, c.traffic, root)

        # Add in Conditionals
        for k, v in mapper.items():
            if k.dim in c.guards:
                node = NodeConditional(c.guards[k.dim])
                v.last.parent = node
                node.parent = v

    return stree


def section(stree):
    """
    Create sections in a :class:`ScheduleTree`. A section is a sub-tree with
    the following properties: ::

        * The root is a node of type :class:`NodeSection`;
        * The immediate children of the root are nodes of type :class:`NodeIteration`
          and have same parent.
        * The :class:`Dimension` of the immediate children are either: ::
            * identical, OR
            * different, but all of type :class:`SubDimension`;
        * The :class:`Dimension` of the immediate children cannot be a
          :class:`TimeDimension`.
    """

    class Section(object):
        def __init__(self, node):
            self.parent = node.parent
            self.dim = node.dim
            self.nodes = [node]

        def is_compatible(self, node):
            return (self.parent == node.parent
                    and (self.dim == node.dim or node.dim.is_Sub))

    # Search candidate sections
    sections = []
    for i in range(stree.height):
        # Find all sections at depth `i`
        section = None
        for n in findall(stree, filter_=lambda n: n.depth == i):
            if any(p in flatten(s.nodes for s in sections) for p in n.ancestors):
                # Already within a section
                continue
            elif not n.is_Iteration or n.dim.is_Time:
                section = None
            elif section is None or not section.is_compatible(n):
                section = Section(n)
                sections.append(section)
            else:
                section.nodes.append(n)

    # Transform the schedule tree by adding in sections
    for i in sections:
        node = NodeSection()
        processed = []
        for n in list(i.parent.children):
            if n in i.nodes:
                n.parent = node
                if node not in processed:
                    processed.append(node)
            else:
                processed.append(n)
        i.parent.children = processed

    return stree

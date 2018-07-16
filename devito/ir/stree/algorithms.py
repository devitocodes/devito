from collections import OrderedDict

from anytree import LevelOrderIter, findall

from devito.ir.stree.tree import (ScheduleTree, NodeIteration, NodeConditional,
                                  NodeExprs, NodeSection, NodeHalo, insert)
from devito.ir.support.space import IterationSpace
from devito.mpi import HaloScheme, HaloSchemeException
from devito.parameters import configuration
from devito.tools import flatten

__all__ = ['st_build']


def st_build(clusters):
    """
    Create a :class:`ScheduleTree` from a :class:`ClusterGroup`.
    """
    # ClusterGroup -> Schedule tree
    stree = st_schedule(clusters)

    # Add in section nodes
    stree = st_section(stree)

    # Add in halo update nodes
    stree = st_make_halo(stree)

    return stree


def st_schedule(clusters):
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
            mapper[i].ispace = IterationSpace.merge(mapper[i].ispace, c.ispace)
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


def st_make_halo(stree):
    """
    Add :class:`NodeHalo` to a :class:`ScheduleTree`. A halo node describes
    what halo exchanges should take place before executing the sub-tree.
    """
    processed = {}
    for n in LevelOrderIter(stree, stop=lambda i: i.parent in processed):
        if not n.is_Iteration:
            continue
        exprs = flatten(i.exprs for i in findall(n, lambda i: i.is_Exprs))
        try:
            halo_scheme = HaloScheme(exprs)
            if n.dim in halo_scheme.dmapper:
                processed[n] = NodeHalo(halo_scheme)
        except HaloSchemeException:
            # We should get here only when trying to compute a halo
            # scheme for a group of expressions that belong to different
            # iteration spaces. We expect proper halo schemes to be built
            # as the `stree` visit proceeds.
            # TODO: However, at the end, we should check that a halo scheme,
            # possibly even a "void" one, has been built for *all* of the
            # expressions, and error out otherwise.
            continue
        except RuntimeError as e:
            if configuration['mpi'] is True:
                raise RuntimeError(str(e))

    for k, v in processed.items():
        insert(v, k.parent, [k])

    return stree


def st_section(stree):
    """
    Add :class:`NodeSection` to a :class:`ScheduleTree`. A section defines a
    sub-tree with the following properties: ::

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
        insert(NodeSection(), i.parent, i.nodes)

    return stree

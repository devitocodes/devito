from collections import OrderedDict

from anytree import findall

from devito.ir.stree.tree import (ScheduleTree, NodeIteration, NodeConditional,
                                  NodeExprs, NodeSection, NodeHalo, insert)
from devito.ir.support import IterationSpace
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
            mapper[i].ispace = IterationSpace.merge(mapper[i].ispace,
                                                    c.ispace.project([i.dim]))
        # Later sub-trees, instead, will not be used anymore
        for i in pointers[index:]:
            mapper.pop(i)

        # Add in Iterations
        for i in c.itintervals[index:]:
            root = NodeIteration(c.ispace.project([i.dim]), root)
            mapper[i] = root

        # Add in Expressions
        NodeExprs(c.exprs, c.ispace, c.dspace, c.shape, c.ops, c.traffic, root)

        # Add in Conditionals
        for k, v in mapper.items():
            if k.dim in c.guards:
                node = NodeConditional(c.guards[k.dim])
                v.last.parent = node
                node.parent = v

    return stree


def st_make_halo(stree):
    """
    Add :class:`NodeHalo`s to a :class:`ScheduleTree`. A HaloNode captures
    the halo exchanges that should take place before executing the sub-tree;
    these are described by means of a :class:`HaloScheme`.
    """
    # Build a HaloScheme for each expression bundle
    halo_schemes = {}
    for n in findall(stree, lambda i: i.is_Exprs):
        try:
            halo_schemes[n] = HaloScheme(n.exprs, n.ispace, n.dspace)
        except HaloSchemeException as e:
            if configuration['mpi']:
                raise RuntimeError(str(e))

    # Insert the HaloScheme at a suitable level in the ScheduleTree
    mapper = {}
    for k, hs in halo_schemes.items():
        for f, v in hs.fmapper.items():
            spot = k
            ancestors = [n for n in k.ancestors if n.is_Iteration]
            for n in ancestors:
                test0 = any(n.dim is i.dim for i in v.halos)
                test1 = n.dim not in [i.root for i in v.loc_indices]
                if test0 or test1:
                    spot = n
                    break
            mapper.setdefault(spot, []).append((f, v))
    for spot, entries in mapper.items():
        insert(NodeHalo(HaloScheme(fmapper=dict(entries))), spot.parent, [spot])

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

from collections import OrderedDict

from anytree import findall

from devito.ir.stree.tree import (ScheduleTree, NodeIteration, NodeConditional,
                                  NodeSync, NodeExprs, NodeSection, NodeHalo, insert)
from devito.ir.support import SEQUENTIAL, IterationSpace, UniteratedInterval
from devito.mpi import HaloScheme, HaloSchemeException
from devito.parameters import configuration
from devito.tools import flatten

__all__ = ['stree_build']


def stree_build(clusters):
    """
    Create a ScheduleTree from a ClusterGroup.
    """
    # ClusterGroup -> ScheduleTree
    stree = stree_schedule(clusters)

    # Add in section nodes
    stree = stree_section(stree)

    # Add in halo update nodes
    stree = stree_make_halo(stree)

    return stree


def stree_schedule(clusters):
    """
    Arrange an iterable of Clusters into a ScheduleTree.
    """
    stree = ScheduleTree()

    mapper = OrderedDict()
    for c in clusters:
        pointers = list(mapper)

        # Find out if any of the existing nodes can be reused
        index = 0
        root = stree
        for it0, it1 in zip(c.itintervals, pointers):
            if it0 != it1:
                break
            root = mapper[it0]
            index += 1
            if it0.dim in c.guards:
                break

        # The reused sub-trees might acquire some new sub-iterators
        for i in pointers[:index]:
            mapper[i].ispace = IterationSpace.union(mapper[i].ispace,
                                                    c.ispace.project([i.dim]))
        # Nested sub-trees, instead, will not be used anymore
        for i in pointers[index:]:
            mapper.pop(i)

        # Add in Iterations
        for i in c.itintervals[index:]:
            root = NodeIteration(c.ispace.project([i.dim]), root, c.properties.get(i.dim))
            mapper[i] = root

        # Add in Expressions
        NodeExprs(c.exprs, c.ispace, c.dspace, c.ops, c.traffic, root)

        # Add in Conditionals and Syncs, which chop down the reuse tree
        drop = None
        for k, v in [(UniteratedInterval, stree)] + list(mapper.items()):
            if drop:
                mapper.pop(k)
            if k.dim in c.syncs:
                node = NodeSync(c.syncs[k.dim])
                v.last.parent = node
                node.parent = v
                drop = True
            if k.dim in c.guards:
                node = NodeConditional(c.guards[k.dim])
                v.last.parent = node
                node.parent = v
                drop = True

    return stree


def stree_make_halo(stree):
    """
    Add NodeHalos to a ScheduleTree. A NodeHalo captures the halo exchanges
    that should take place before executing the sub-tree; these are described
    by means of a HaloScheme.
    """
    # Build a HaloScheme for each expression bundle
    halo_schemes = {}
    for n in findall(stree, lambda i: i.is_Exprs):
        try:
            halo_schemes[n] = HaloScheme(n.exprs, n.ispace)
        except HaloSchemeException as e:
            if configuration['mpi']:
                raise RuntimeError(str(e))

    # Split a HaloScheme based on where it should be inserted
    # For example, it's possible that, for a given HaloScheme, a Function's
    # halo needs to be exchanged at a certain `stree` depth, while another
    # Function's halo needs to be exchanged before some other nodes
    mapper = {}
    for k, hs in halo_schemes.items():
        for f, v in hs.fmapper.items():
            spot = k
            ancestors = [n for n in k.ancestors if n.is_Iteration]
            for n in ancestors:
                # Place the halo exchange right before the first
                # distributed Dimension which requires it
                if any(i.dim in n.dim._defines for i in v.halos):
                    spot = n
                    break
            mapper.setdefault(spot, []).append(hs.project(f))

    # Now fuse the HaloSchemes at the same `stree` depth and perform the insertion
    for spot, halo_schemes in mapper.items():
        insert(NodeHalo(HaloScheme.union(halo_schemes)), spot.parent, [spot])

    return stree


def stree_section(stree):
    """
    Add NodeSections to a ScheduleTree. A NodeSection, or simply "section",
    defines a sub-tree with the following properties:

        * The root is a node of type NodeSection;
        * The immediate children of the root are nodes of type NodeIteration;
        * The Dimensions of the immediate children are either:
            * identical, OR
            * different, but all of type SubDimension;
        * The Dimension of the immediate children cannot be a TimeDimension.
    """

    class Section(object):
        def __init__(self, node):
            self.parent = node.parent
            try:
                self.dim = node.dim
            except AttributeError:
                self.dim = None
            self.nodes = [node]

        def is_compatible(self, node):
            return self.parent == node.parent and self.dim.root == node.dim.root

    # Search candidate sections
    sections = []
    for i in range(stree.height):
        # Find all sections at depth `i`
        section = None
        for n in findall(stree, filter_=lambda n: n.depth == i):
            if any(p in flatten(s.nodes for s in sections) for p in n.ancestors):
                # Already within a section
                continue
            elif n.is_Sync:
                # SyncNodes are self-contained
                sections.append(Section(n))
                section = None
            elif n.is_Iteration:
                if n.dim.is_Time and SEQUENTIAL in n.properties:
                    # If n.dim.is_Time, we end up here in 99.9% of the cases.
                    # Sometimes, however, time is a PARALLEL Dimension (e.g.,
                    # think of `norm` Operators)
                    section = None
                elif section is None or not section.is_compatible(n):
                    section = Section(n)
                    sections.append(section)
                else:
                    section.nodes.append(n)
            else:
                section = None

    # Transform the schedule tree by adding in sections
    for i in sections:
        insert(NodeSection(), i.parent, i.nodes)

    return stree

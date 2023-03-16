from collections import defaultdict
from itertools import groupby

from anytree import findall
from sympy import And

from devito.ir.clusters import Cluster
from devito.ir.stree.tree import (ScheduleTree, NodeIteration, NodeConditional,
                                  NodeSync, NodeExprs, NodeSection, NodeHalo)
from devito.ir.support import (SEQUENTIAL, Any, Interval, IterationInterval,
                               IterationSpace, normalize_properties)
from devito.mpi.halo_scheme import HaloScheme
from devito.tools import Bunch, DefaultOrderedDict
from devito.types.dimension import BOTTOM

__all__ = ['stree_build']


def stree_build(clusters, profiler=None, **kwargs):
    """
    Create a ScheduleTree from a ClusterGroup.
    """
    clusters, hsmap = preprocess(clusters)

    stree = ScheduleTree()
    section = None

    base = IterationInterval(Interval(BOTTOM), [], Any)
    prev = Cluster(None)

    mapper = DefaultOrderedDict(lambda: Bunch(top=None, bottom=None))
    mapper[base] = Bunch(top=stree, bottom=stree)

    for c in clusters:
        if reuse_subtree(c, prev):
            tip = mapper[base].bottom
            maybe_reusable = prev.itintervals
        else:
            # Add any guards/Syncs outside of the outermost Iteration
            tip = augment_subtree(c, None, stree)
            maybe_reusable = []

        # Is there a HaloTouch to attach?
        try:
            hs = hsmap[c]
        except KeyError:
            hs = None

        index = 0
        for it0, it1 in zip(c.itintervals, maybe_reusable):
            if it0 != it1:
                break

            d = it0.dim
            if needs_nodehalo(d, hs):
                break

            index += 1

            # The reused sub-trees might acquire new sub-iterators as well as
            # new properties
            mapper[it0].top.ispace = IterationSpace.union(
                mapper[it0].top.ispace, c.ispace.project([d])
            )
            mapper[it0].top.properties = normalize_properties(
                mapper[it0].top.properties, c.properties[it0.dim]
            )

            if reuse_subtree(c, prev, d):
                tip = mapper[it0].bottom
            else:
                tip = mapper[it0].top
                tip = augment_subtree(c, d, tip)
                mapper[it0].bottom = tip
                break

        # Nested sub-trees, instead, will not be used anymore
        for it in prev.itintervals[index:]:
            mapper.pop(it)
        prev = c

        # Add in Node{Iteration,Conditional,Sync}
        for it in c.itintervals[index:]:
            d = it.dim
            tip = NodeIteration(c.ispace.project([d]), tip, c.properties.get(d, ()))
            mapper[it].top = tip
            tip = augment_subtree(c, d, tip)
            mapper[it].bottom = tip

        # Attach NodeHalo if necessary
        for it, v in mapper.items():
            if needs_nodehalo(it.dim, hs):
                v.bottom.parent = NodeHalo(hs, v.bottom.parent)
                break

        # Add in NodeExprs
        exprs = []
        for conditionals, g in groupby(c.exprs, key=lambda e: e.conditionals):
            exprs = list(g)

            # Indirect ConditionalDimensions induce expression-level guards
            if conditionals:
                guard = And(*conditionals.values(), evaluate=False)
                parent = NodeConditional(guard, tip)
            else:
                parent = tip

            NodeExprs(exprs, c.ispace, c.dspace, c.ops, c.traffic, parent)

        # Nest within a NodeSection if possible
        if profiler is None or \
           any(i.is_Section for i in reversed(tip.ancestors)):
            continue

        candidate = None
        for i in reversed(tip.ancestors + (tip,)):
            if i.is_Halo:
                candidate = i
            elif i.is_Sync:
                attach_section(i)
                section = None
                break
            elif i.is_Iteration:
                if (i.dim.is_Time and SEQUENTIAL in i.properties):
                    section = attach_section(candidate, section)
                    break
                else:
                    candidate = i
        else:
            attach_section(candidate, section)

    return stree


# *** Utility functions to construct the ScheduleTree


def preprocess(clusters):
    """
    Remove the HaloTouches from `clusters` and create a mapping associating
    each removed HaloTouch to the first Cluster necessitating it.
    """
    processed = []
    hsmap = defaultdict(list)

    queue = []

    for c in clusters:
        if c.is_halo_touch:
            queue.append(HaloScheme.union(e.rhs.halo_scheme for e in c.exprs))
        else:
            dims = set(c.ispace.promote(lambda d: d.is_Block).itdimensions)

            for hs in list(queue):
                if hs.distributed_aindices & dims:
                    queue.remove(hs)
                    hsmap[c].append(hs)

            processed.append(c)

    hsmap = {c: HaloScheme.union(hss) for c, hss in hsmap.items()}

    return processed, hsmap


def reuse_subtree(c0, c1, d=None):
    return (c0.guards.get(d) == c1.guards.get(d) and
            c0.syncs.get(d) == c1.syncs.get(d))


def augment_subtree(cluster, d, tip):
    if d in cluster.guards:
        tip = NodeConditional(cluster.guards[d], tip)
    if d in cluster.syncs:
        tip = NodeSync(cluster.syncs[d], tip)
    return tip


def needs_nodehalo(d, hs):
    return hs and d._defines.intersection(hs.distributed_aindices)


def reuse_section(candidate, section):
    try:
        if not section or candidate.siblings[-1] is not section:
            return False
    except IndexError:
        return False

    key = lambda i: findall(i, lambda n: n.is_Iteration)
    iters0 = key(section)
    iters1 = key(candidate)

    # Heuristics for NodeSection reuse:
    # * Ignore some kinds of iteration Dimensions
    key = lambda d: not (d.is_Custom or d.is_Stencil)
    iters0 = [i for i in iters0 if key(i.dim)]
    iters1 = [i for i in iters1 if key(i.dim)]

    # * Same set of iteration Dimensions
    key = lambda i: i.interval.promote(lambda d: d.is_Block).dim
    test00 = len(iters0) == len(iters1)
    test01 = all(key(i) is key(j) for i, j in zip(iters0, iters1))

    # * All subtrees use at least one local SubDimension (i.e., BCs)
    key = lambda iters: any(i.dim.is_Sub and i.dim.local for i in iters)
    test1 = key(iters0) and key(iters1)

    if (test00 and test01) or test1:
        candidate.parent = section
        return True
    else:
        return False


def attach_section(candidate, section=None):
    if candidate and not reuse_section(candidate, section):
        section = NodeSection()
        section.parent = candidate.parent
        candidate.parent = section

    return section

from itertools import groupby

from anytree import findall
from sympy import And

from devito.ir.clusters import Cluster
from devito.ir.stree.tree import (ScheduleTree, NodeIteration, NodeConditional,
                                  NodeSync, NodeExprs, NodeSection, NodeHalo)
from devito.ir.support import (SEQUENTIAL, Any, Interval, IterationInterval,
                               IterationSpace, normalize_properties, normalize_syncs)
from devito.mpi.halo_scheme import HaloScheme
from devito.tools import Bunch, DefaultOrderedDict

__all__ = ['stree_build']


def stree_build(clusters, profiler=None, **kwargs):
    """
    Create a ScheduleTree from a ClusterGroup.
    """
    clusters = preprocess(clusters, **kwargs)

    stree = ScheduleTree()
    section = None

    prev = Cluster(None)

    mapper = DefaultOrderedDict(lambda: Bunch(top=None, middle=None, bottom=None))
    mapper[base] = Bunch(top=stree, middle=stree, bottom=stree)

    for c in clusters:
        if reuse_whole_subtree(c, prev):
            tip = mapper[base].bottom
            maybe_reusable = prev.itintervals
        else:
            # Add any guards/Syncs outside of the outermost Iteration
            tip = augment_whole_subtree(c, stree, mapper, base)
            maybe_reusable = []

        index = 0
        for it0, it1 in zip(c.itintervals, maybe_reusable):
            if it0 != it1:
                break

            d = it0.dim
            if needs_nodehalo(d, c.halo_scheme):
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

            if reuse_whole_subtree(c, prev, d):
                tip = mapper[it0].bottom
            elif reuse_partial_subtree(c, prev, d):
                tip = mapper[it0].middle
                tip = augment_partial_subtree(c, tip, mapper, it0)
                break
            else:
                tip = mapper[it0].top
                tip = augment_whole_subtree(c, tip, mapper, it0)
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
            tip = augment_whole_subtree(c, tip, mapper, it)

        # Attach NodeHalo if necessary
        for it, v in mapper.items():
            if needs_nodehalo(it.dim, c.halo_scheme):
                v.bottom.parent = NodeHalo(c.halo_scheme, v.bottom.parent)
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


# *** Utilities to construct the ScheduleTree

base = IterationInterval(Interval(None), [], Any)


def preprocess(clusters, options=None, **kwargs):
    """
    Remove the HaloTouch's from `clusters` and create a mapping associating
    each removed HaloTouch to the first Cluster necessitating it.
    """
    queue = []
    processed = []
    for c in clusters:
        if c.is_halo_touch:
            hs = HaloScheme.union(e.rhs.halo_scheme for e in c.exprs)
            queue.append(c.rebuild(halo_scheme=hs))
        else:
            dims = set(c.ispace.promote(lambda d: d.is_Block).itdims)

            found = []
            for c1 in list(queue):
                distributed_aindices = c1.halo_scheme.distributed_aindices
                h_indices = set().union(*[d._defines for d in c1.halo_scheme.loc_indices])

                # Skip if the halo exchange would end up outside
                # its iteration space
                if h_indices and not h_indices & dims:
                    continue

                diff = dims - distributed_aindices
                intersection = dims & distributed_aindices

                if all(c1.guards.get(d) == c.guards.get(d) for d in diff) and \
                   len(intersection) > 0:
                    found.append(c1)
                    queue.remove(c1)

            syncs = normalize_syncs(*[c1.syncs for c1 in found])
            if syncs:
                ispace = c.ispace.project(syncs)
                processed.append(c.rebuild(exprs=[], ispace=ispace, syncs=syncs))

            halo_scheme = HaloScheme.union([c1.halo_scheme for c1 in found])
            processed.append(c.rebuild(halo_scheme=halo_scheme))

    # Sanity check!
    try:
        assert not queue
    except AssertionError:
        if options['mpi']:
            raise RuntimeError("Unsupported MPI for the given equations")

    return processed


def reuse_partial_subtree(c0, c1, d=None):
    return c0.guards.get(d) == c1.guards.get(d)


def reuse_whole_subtree(c0, c1, d=None):
    return (c0.guards.get(d) == c1.guards.get(d) and
            c0.syncs.get(d) == c1.syncs.get(d))


def augment_partial_subtree(cluster, tip, mapper, it=None):
    d = it.dim

    if d in cluster.syncs:
        tip = NodeSync(cluster.syncs[d], tip)

    mapper[it].bottom = tip

    return tip


def augment_whole_subtree(cluster, tip, mapper, it):
    d = it.dim

    if d in cluster.guards:
        tip = NodeConditional(cluster.guards[d], tip)

    mapper[it].middle = mapper[it].bottom = tip

    return augment_partial_subtree(cluster, tip, mapper, it)


def needs_nodehalo(d, hs):
    return d and hs and d._defines.intersection(hs.distributed_aindices)


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

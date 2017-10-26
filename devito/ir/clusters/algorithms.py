from collections import OrderedDict, namedtuple

from devito.ir.clusters.cluster import Cluster
from devito.ir.dfg import temporaries_graph
from devito.ir.support import Stencil
from devito.symbolics import xreplace_indices
from devito.types import Scalar

__all__ = ['clusterize', 'optimize']


def merge(clusters):
    """
    Given an ordered collection of :class:`Cluster` objects, return a
    (potentially) smaller sequence in which clusters with identical stencil
    have been merged into a single :class:`Cluster`.
    """
    mapper = OrderedDict()
    for c in clusters:
        mapper.setdefault((c.stencil.entries, c.atomics), []).append(c)

    processed = []
    for (entries, atomics), clusters in mapper.items():
        # Eliminate redundant temporaries
        temporaries = OrderedDict()
        for c in clusters:
            for k, v in c.trace.items():
                if k not in temporaries:
                    temporaries[k] = v
        # Squash the clusters together
        processed.append(Cluster(temporaries.values(), Stencil(entries), atomics))

    return processed


def optimize(clusters):
    """
    Attempt scalar promotion. Candidates are tensors that do not appear in
    any other clusters.
    """
    clusters = merge(clusters)

    processed = []
    for c1 in clusters:
        mapper = {}
        temporaries = []
        for k, v in c1.trace.items():
            if v.function.is_Array and\
                    not any(v.function in c2.unknown for c2 in clusters):
                for i in c1.tensors[v.function]:
                    # LHS scalarization
                    scalarized = Scalar(name='s%d' % len(mapper)).indexify()
                    mapper[i] = scalarized

                    # May have to "unroll" some tensor expressions for scalarization;
                    # e.g., if we have two occurrences of r0, say r0[x,y,z] and
                    # r0[x+1,y,z], and r0 is to be scalarized, this will require a
                    # different scalar for each unique set of indices.
                    assert len(v.function.indices) == len(k.indices) == len(i.indices)
                    shifting = {idx: idx + (o2 - o1) for idx, o1, o2 in
                                zip(v.function.indices, k.indices, i.indices)}

                    # Transform /v/, introducing (i) a scalarized LHS and (ii) shifted
                    # indices if necessary
                    handle = v.func(scalarized, v.rhs.xreplace(mapper))
                    handle = xreplace_indices(handle, shifting)
                    temporaries.append(handle)
            else:
                temporaries.append(v.func(k, v.rhs.xreplace(mapper)))
        processed.append(c1.rebuild(temporaries))

    return processed


def aggregate(exprs, stencils):
    """
    Aggregate consecutive expressions having identical LHS and identical
    stencil by substituting the earlier with the later ones, in program order.

    Examples
    ========
    a[i] = b[i] + 3.
    a[i] = c[i] + 4. + a[i]
    >> a[i] = c[i] + 4. + b[i] + 3.
    """
    mapper = OrderedDict(zip(exprs, stencils))

    groups = []
    last = None
    for k, v in mapper.items():
        key = k.lhs, v
        if key == last:
            groups[-1].append(k)
        else:
            last = key
            groups.append([k])

    exprs, stencils = [], []
    for i in groups:
        top = i[0]
        if len(i) == 1:
            exprs.append(top)
        else:
            inlining, base = top.args
            queue = list(i[1:])
            while queue:
                base = queue.pop(0).rhs.xreplace({inlining: base})
            exprs.append(top.func(inlining, base))
        stencils.append(mapper[top])

    return exprs, stencils


def clusterize(exprs, stencils, atomics=None):
    """
    Derive :class:`Cluster` objects from an iterable of expressions; a stencil for
    each expression must be provided. A list of atomic dimensions (see description
    in Cluster.__doc__) may be provided.
    """
    assert len(exprs) == len(stencils)

    exprs, stencils = aggregate(exprs, stencils)

    Info = namedtuple('Info', 'trace stencil')

    g = temporaries_graph(exprs)
    mapper = OrderedDict([(k, Info(g.trace(k) + g.trace(k, readby=True, strict=True), j))
                          for (k, v), j in zip(g.items(), stencils) if v.is_tensor])

    # A cluster stencil is determined iteratively, by first calculating the
    # "local" stencil and then by looking at the stencils of all other clusters
    # depending on it. The stencil information is propagated until there are
    # no more updates.
    queue = list(mapper)
    while queue:
        target = queue.pop(0)

        info = mapper[target]
        strict_trace = [i.lhs for i in info.trace if i.lhs != target]

        stencil = Stencil(info.stencil.entries)
        for i in strict_trace:
            if i in mapper:
                stencil = stencil.add(mapper[i].stencil)

        mapper[target] = Info(info.trace, stencil)

        if stencil != info.stencil:
            # Something has changed, need to propagate the update
            queue.extend([i for i in strict_trace if i not in queue])

    clusters = []
    for target, info in mapper.items():
        # Drop all non-output tensors, as computed by other clusters
        exprs = [i for i in info.trace if i.lhs.is_Symbol or i.lhs == target]

        # Create and track the cluster
        clusters.append(Cluster(exprs, info.stencil.frozen, atomics))

    return merge(clusters)

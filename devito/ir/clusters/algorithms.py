from collections import OrderedDict

from devito.ir.support import Scope
from devito.ir.clusters.cluster import PartialCluster, ClusterGroup
from devito.ir.dfg import TemporariesGraph
from devito.symbolics import xreplace_indices
from devito.types import Scalar
from devito.tools import flatten

__all__ = ['clusterize', 'groupby']


def groupby(clusters):
    """
    Given an ordered collection of :class:`PartialCluster` objects, return a
    (potentially) smaller sequence in which dependence-free PartialClusters with
    identical stencil have been squashed into a single PartialCluster.
    """
    clusters = clusters.unfreeze()

    # Attempt cluster fusion
    processed = ClusterGroup()
    for c in clusters:
        fused = False
        for candidate in reversed(list(processed)):
            # Check all data dependences relevant for cluster fusion
            scope = Scope(exprs=candidate.exprs + c.exprs)
            anti = scope.d_anti.carried() - scope.d_anti.increment
            flow = scope.d_flow - (scope.d_flow.inplace() + scope.d_flow.increment)
            funcs = [i.function for i in anti]
            if candidate.stencil == c.stencil and\
                    all(is_local(i, candidate, c, clusters) for i in funcs):
                # /c/ will be fused into /candidate/. All fusion-induced anti
                # dependences are eliminated through so called "index bumping and
                # array contraction", which transforms array accesses into scalars

                # Optimization: we also bump-and-contract the Arrays inducing
                # non-carried dependences, to avoid useless memory accesses
                funcs += [i.function for i in scope.d_flow.independent()
                          if is_local(i.function, candidate, c, clusters)]

                bump_and_contract(funcs, candidate, c)
                candidate.squash(c)
                fused = True
                break
            elif anti:
                # Data dependences prevent fusion with earlier clusters, so
                # must break up the search
                processed.atomics[c].update(set(anti.cause))
                break
            elif set(flow).intersection(processed.atomics[candidate]):
                # We cannot even attempt fusing with earlier clusters, as
                # otherwise the existing flow dependences wouldn't be honored
                break
        # Fallback
        if not fused:
            processed.append(c)

    return processed.freeze()


def is_local(array, source, sink, context):
    """
    Return True if ``array`` satisfies the following conditions: ::

        * it's a temporary; that is, of type :class:`Array`;
        * it's written once, within the ``source`` :class:`PartialCluster`, and
          its value only depends on global data;
        * it's read in the ``sink`` :class:`PartialCluster` only; in particular,
          it doesn't appear in any other :class:`PartialCluster`s out of those
          provided in ``context``.

    If any of these conditions do not hold, return False.
    """
    if not array.is_Array:
        return False

    # Written in source
    written_once = False
    for i in source.trace.values():
        if array == i.function:
            if written_once is True:
                # Written more than once, break
                written_once = False
                break
            reads = [j.base.function for j in i.reads]
            if any(j.is_SymbolicFunction or j.is_Scalar for j in reads):
                # Can't guarantee its value only depends on local data
                written_once = False
                break
            written_once = True
    if written_once is False:
        return False

    # Never read outside of sink
    context = [i for i in context if i not in [source, sink]]
    if array in flatten(i.unknown for i in context):
        return False

    return True


def bump_and_contract(targets, source, sink):
    """
    Transform in-place the PartialClusters ``source`` and ``sink`` by turning the
    :class:`Array`s in ``targets`` into :class:`Scalar`. This is implemented
    through index bumping and array contraction.

    :param targets: The :class:`Array` objects that will be contracted.
    :param source: The source :class:`PartialCluster`.
    :param sink: The sink :class:`PartialCluster`.

    Examples
    ========
    Index bumping
    -------------
    Given: ::

        r[x,y,z] = b[x,y,z]*2

    Produce: ::

        r[x,y,z] = b[x,y,z]*2
        r[x,y,z+1] = b[x,y,z+1]*2

    Array contraction
    -----------------
    Given: ::

        r[x,y,z] = b[x,y,z]*2
        r[x,y,z+1] = b[x,y,z+1]*2

    Produce: ::

        tmp0 = b[x,y,z]*2
        tmp1 = b[x,y,z+1]*2

    Full example (bump+contraction)
    -------------------------------
    Given: ::

        source: [r[x,y,z] = b[x,y,z]*2]
        sink: [a = ... r[x,y,z] ... r[x,y,z+1] ...]
        targets: r

    Produce: ::

        source: [tmp0 = b[x,y,z]*2, tmp1 = b[x,y,z+1]*2]
        sink: [a = ... tmp0 ... tmp1 ...]
    """
    if not targets:
        return

    mapper = {}

    # source
    processed = []
    for k, v in source.trace.items():
        if any(v.function not in i for i in [targets, sink.tensors]):
            processed.append(v.func(k, v.rhs.xreplace(mapper)))
        else:
            for i in sink.tensors[v.function]:
                scalarized = Scalar(name='s%d' % len(mapper)).indexify()
                mapper[i] = scalarized

                # Index bumping
                assert len(v.function.indices) == len(k.indices) == len(i.indices)
                shifting = {idx: idx + (o2 - o1) for idx, o1, o2 in
                            zip(v.function.indices, k.indices, i.indices)}

                # Array contraction
                handle = v.func(scalarized, v.rhs.xreplace(mapper))
                handle = xreplace_indices(handle, shifting)
                processed.append(handle)
    source.exprs = processed

    # sink
    processed = [v.func(k, v.rhs.xreplace(mapper)) for k, v in sink.trace.items()]
    sink.exprs = processed


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


def clusterize(exprs, stencils):
    """
    Derive :class:`Cluster` objects from an iterable of expressions; a stencil for
    each expression must be provided.
    """
    assert len(exprs) == len(stencils)

    exprs, stencils = aggregate(exprs, stencils)

    # Create a PartialCluster for each sequence of expressions computing a tensor
    mapper = OrderedDict()
    g = TemporariesGraph(exprs)
    for (k, v), j in zip(g.items(), stencils):
        if v.is_tensor:
            exprs = g.trace(k)
            exprs += tuple(i for i in g.trace(k, readby=True) if i not in exprs)
            mapper[k] = PartialCluster(exprs, j)

    # Update the PartialClusters' Stencils by looking at the Stencil of the
    # surrounding PartialClusters.
    queue = list(mapper)
    while queue:
        target = queue.pop(0)

        pc = mapper[target]
        strict_trace = [i.lhs for i in pc.exprs if i.lhs != target]

        stencil = pc.stencil.copy()
        for i in strict_trace:
            if i in mapper:
                stencil = stencil.add(mapper[i].stencil)

        if stencil != pc.stencil:
            # Something has changed, need to propagate the update
            pc.stencil = stencil
            queue.extend([i for i in strict_trace if i not in queue])

    # Drop all non-output tensors, as computed by other clusters
    clusters = ClusterGroup()
    for target, pc in mapper.items():
        exprs = [i for i in pc.exprs if i.lhs.is_Symbol or i.lhs == target]
        clusters.append(PartialCluster(exprs, pc.stencil))

    # Attempt grouping as many PartialClusters as possible together
    return groupby(clusters)

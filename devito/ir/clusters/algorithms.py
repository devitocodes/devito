from collections import OrderedDict

from devito.ir.support import Scope
from devito.ir.clusters.cluster import PartialCluster, ClusterGroup
from devito.symbolics import xreplace_indices
from devito.types import Scalar
from devito.tools import Bunch, flatten

__all__ = ['clusterize', 'groupby']


def groupby(clusters):
    """
    Attempt grouping :class:`PartialCluster`s together to create bigger
    :class:`PartialCluster`s (i.e., containing more expressions).

    .. note::

        This function relies on advanced data dependency analysis tools
        based upon classic Lamport theory.
    """
    clusters = clusters.unfreeze()

    processed = ClusterGroup()
    for c in clusters:
        fused = False
        for candidate in reversed(list(processed)):
            # Collect all relevant data dependences
            scope = Scope(exprs=candidate.exprs + c.exprs)
            anti = scope.d_anti.carried() - scope.d_anti.increment
            flow = scope.d_flow - (scope.d_flow.inplace() + scope.d_flow.increment)
            funcs = [i.function for i in anti]
            if candidate.ispace == c.ispace and\
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

    return processed


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


def clusterize(exprs):
    """Group a sequence of :class:`ir.Eq`s into one or more :class:`Cluster`s."""

    # Build a graph capturing the dependencies among the input tensor expressions
    mapper = OrderedDict()
    for i, e1 in enumerate(exprs):
        trace = [e2 for e2 in exprs[:i] if Scope([e2, e1]).has_dep] + [e1]
        trace.extend([e2 for e2 in exprs[i+1:] if Scope([e1, e2]).has_dep])
        mapper[e1] = Bunch(trace=trace, ispace=e1.ispace)

    # Derive the iteration spaces
    queue = list(mapper)
    while queue:
        target = queue.pop(0)

        ispaces = [mapper[i].ispace for i in mapper[target].trace]

        coerced_ispace = mapper[target].ispace.intersection(*ispaces)

        if coerced_ispace != mapper[target].ispace:
            # Something has changed, need to propagate the update
            mapper[target].ispace = coerced_ispace
            queue.extend([i for i in mapper[target].trace if i not in queue])

    # Build a PartialCluster for each tensor expression
    clusters = ClusterGroup()
    for k, v in mapper.items():
        if k.is_Tensor:
            scalars = [i for i in v.trace[:v.trace.index(k)] if i.is_Scalar]
            clusters.append(PartialCluster(scalars + [k], v.ispace))

    # Group PartialClusters together where possible
    clusters = groupby(clusters)

    return clusters.finalize()

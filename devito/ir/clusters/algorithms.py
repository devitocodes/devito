from collections import OrderedDict

from devito.ir.support import (Scope, IterationSpace, detect_flow_directions,
                               force_directions, group_expressions)
from devito.ir.clusters.cluster import PartialCluster, ClusterGroup
from devito.symbolics import CondEq, xreplace_indices
from devito.types import Scalar
from devito.tools import filter_sorted, flatten

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
            # Guarded clusters cannot be grouped together
            if c.guards or candidate.guards:
                break

            # Collect all relevant data dependences
            scope = Scope(exprs=candidate.exprs + c.exprs)

            # Collect anti-dependences preventing grouping
            anti = scope.d_anti.carried() - scope.d_anti.increment
            funcs = [i.function for i in anti]

            # Collect flow-dependences breaking the search
            flow = scope.d_flow - (scope.d_flow.inplace() + scope.d_flow.increment)

            if candidate.ispace.is_compatible(c.ispace) and\
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
                c.atomics.update(anti.cause)
                break
            elif flow.cause & candidate.atomics:
                # We cannot even attempt fusing with earlier clusters, as
                # otherwise the existing flow dependences wouldn't be honored
                break
        # Fallback
        if not fused:
            processed.append(c)

    return processed


def guard(clusters):
    """
    Return a new :class:`ClusterGroup` with a new :class:`PartialCluster`
    for each conditional expression encountered in ``clusters``.
    """
    processed = ClusterGroup()
    for c in clusters:
        # Separate the expressions that should be guarded from the free ones
        mapper = OrderedDict()
        free = []
        for e in c.exprs:
            if e.conditionals:
                mapper.setdefault(tuple(filter_sorted(e.conditionals)), []).append(e)
            else:
                free.append(e)

        # Some expressions may not require guards at all. We put them in their
        # own cluster straight away
        if free:
            processed.append(PartialCluster(free, c.ispace, c.dspace, c.atomics))

        # Then we add in all guarded clusters
        for k, v in mapper.items():
            guards = {d.parent: d.condition or CondEq(d.parent % d.factor, 0) for d in k}
            processed.append(PartialCluster(v, c.ispace, c.dspace, c.atomics, guards))

    return ClusterGroup(processed)


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
            if any(j.is_TensorFunction or j.is_Scalar for j in reads):
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
    for e in source.exprs:
        function = e.lhs.base.function
        if any(function not in i for i in [targets, sink.tensors]):
            processed.append(e.func(e.lhs, e.rhs.xreplace(mapper)))
        else:
            for i in sink.tensors[function]:
                scalar = Scalar(name='s%s%d' % (i.function.name, len(mapper))).indexify()
                mapper[i] = scalar

                # Index bumping
                assert len(function.indices) == len(e.lhs.indices) == len(i.indices)
                shifting = {idx: idx + (o2 - o1) for idx, o1, o2 in
                            zip(function.indices, e.lhs.indices, i.indices)}

                # Array contraction
                handle = e.func(scalar, e.rhs.xreplace(mapper))
                handle = xreplace_indices(handle, shifting)
                processed.append(handle)
    source.exprs = processed

    # sink
    processed = [e.func(e.lhs, e.rhs.xreplace(mapper)) for e in sink.exprs]
    sink.exprs = processed


def clusterize(exprs):
    """Group a sequence of :class:`ir.Eq`s into one or more :class:`Cluster`s."""
    # Group expressions based on data dependences
    groups = group_expressions(exprs)

    clusters = ClusterGroup()

    # Coerce iteration direction of each expression in each group
    for g in groups:
        mapper = OrderedDict([(e, e.directions) for e in g])
        flowmap = detect_flow_directions(g)
        queue = list(g)
        while queue:
            k = queue.pop(0)
            directions, _ = force_directions(flowmap, lambda i: mapper[k].get(i))
            directions = {i: directions[i] for i in mapper[k]}
            # Need update propagation ?
            if directions != mapper[k]:
                mapper[k] = directions
                queue.extend([i for i in g if i not in queue])

        # Wrap each tensor expression in a PartialCluster
        for k, v in mapper.items():
            if k.is_Tensor:
                scalars = [i for i in g[:g.index(k)] if i.is_Scalar]
                intervals, sub_iterators, _ = k.ispace.args
                ispace = IterationSpace(intervals, sub_iterators, v)
                clusters.append(PartialCluster(scalars + [k], ispace, k.dspace))

    # Group PartialClusters together where possible
    clusters = groupby(clusters)

    # Introduce conditional PartialClusters
    clusters = guard(clusters)

    return clusters.finalize()

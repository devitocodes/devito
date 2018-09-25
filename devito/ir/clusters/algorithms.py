import sympy

from devito.ir.support import (Scope, DataSpace, IterationSpace, detect_flow_directions,
                               force_directions, group_expressions)
from devito.ir.clusters.cluster import PartialCluster, ClusterGroup
from devito.symbolics import CondEq, xreplace_indices
from devito.types import Scalar
from devito.tools import flatten

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
            if c.guards:
                break

            # Collect all relevant data dependences
            scope = Scope(exprs=candidate.exprs + c.exprs)

            # Collect anti-dependences preventing grouping
            anti = scope.d_anti.carried() - scope.d_anti.increment
            funcs = set(anti.functions)

            # Collect flow-dependences breaking the search
            flow = scope.d_flow - (scope.d_flow.inplace() + scope.d_flow.increment)

            # Can we group `c` with `candidate`?
            test0 = not candidate.guards  # No intervening guards
            test1 = candidate.ispace.is_compatible(c.ispace)  # Compatible ispaces
            test2 = all(is_local(i, candidate, c, clusters) for i in funcs)  # No antideps
            if test0 and test1 and test2:
                # Yes, `c` can be grouped with `candidate`. All anti-dependences
                # (if any) can be eliminated through "index bumping and array
                # contraction", which turns Array temporaries into Scalar temporaries

                # Optimization: we also bump-and-contract the Arrays inducing
                # non-carried dependences, to minimize the working set
                funcs.update({i.function for i in scope.d_flow.independent()
                              if is_local(i.function, candidate, c, clusters)})

                bump_and_contract(funcs, candidate, c)
                candidate.squash(c)
                fused = True
                break
            elif anti:
                # Data dependences prevent fusion with earlier Clusters, so
                # must break up the search
                c.atomics.update(anti.cause)
                break
            elif flow.cause & candidate.atomics:
                # We cannot even attempt fusing with earlier Clusters, as
                # otherwise the carried flow dependences wouldn't be honored
                break
            elif set(candidate.guards) & set(c.dimensions):
                # Like above, we can't attempt fusion with earlier Clusters.
                # Time time because there are intervening conditionals along
                # one or more of the shared iteration dimensions
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
        free = []
        for e in c.exprs:
            if e.conditionals:
                # Expressions that need no guarding are kept in a separate Cluster
                if free:
                    processed.append(PartialCluster(free, c.ispace, c.dspace, c.atomics))
                    free = []
                # Create a guarded PartialCluster
                guards = {}
                for d in e.conditionals:
                    condition = guards.setdefault(d.parent, [])
                    condition.append(d.condition or CondEq(d.parent % d.factor, 0))
                guards = {k: sympy.And(*v, evaluate=False) for k, v in guards.items()}
                processed.append(PartialCluster(e, c.ispace, c.dspace, c.atomics, guards))
            else:
                free.append(e)
        # Leftover
        if free:
            processed.append(PartialCluster(free, c.ispace, c.dspace, c.atomics))

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
    """
    Group a sequence of :class:`ir.Eq`s into one or more :class:`Cluster`s.
    """
    clusters = ClusterGroup()
    for group in group_expressions(exprs):
        flowmap = detect_flow_directions(group)
        prev = None
        for idx, e in enumerate(group):
            if e.is_Tensor:
                scalars = [i for i in group[prev:idx] if i.is_Scalar]
                # Iteration space
                ispace = IterationSpace.merge(e.ispace, *[i.ispace for i in scalars])
                # Enforce iteration directions
                fdirs, _ = force_directions(flowmap, lambda d: ispace.directions.get(d))
                ispace = IterationSpace(ispace.intervals, ispace.sub_iterators, fdirs)
                # Data space
                dspace = DataSpace.merge(e.dspace, *[i.dspace for i in scalars])
                # Prepare for next range
                prev = idx

                clusters.append(PartialCluster(scalars + [e], ispace, dspace))

    # Group PartialClusters together where possible
    clusters = groupby(clusters)

    # Introduce conditional PartialClusters
    clusters = guard(clusters)

    return clusters.finalize()

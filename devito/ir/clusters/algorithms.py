from itertools import chain

import sympy

from devito.ir.support import (Scope, IterationSpace, detect_flow_directions,
                               force_directions)
from devito.ir.clusters.cluster import PartialCluster, Cluster, ClusterGroup
from devito.symbolics import CondEq
from devito.types import Dimension

__all__ = ['clusterize', 'reschedule']


def lift(clusters):
    """
    PartialClusters may be invariant in one or more Dimensions; lifting
    will remove such Dimensions while honoring the topological ordering.
    """
    processed = []
    for i, candidate in enumerate(clusters):
        # Start assuming `candidate` is liftable
        legal = True

        # Safety checks: no reductions, no conditionals, no scalar temporaries
        if any(e.is_Increment for e in candidate.exprs) or\
                candidate.guards or\
                any(e.is_Scalar for e in candidate.exprs):
            legal = False

        # It is a lifting candidate if it redundantly computes values,
        # that is if the iteration space contains more Dimensions than
        # strictly needed (as there are no binding data dependences)
        symbols = set().union(*[e.free_symbols for e in candidate.exprs])
        used_iterators = set().union(*[i._defines for i in symbols
                                        if isinstance(i, Dimension)])
        maybe_liftable = set(candidate.ispace.dimensions) - used_iterators
        if not maybe_liftable:
            legal = False

        # Would the data dependences be honored?
        for c in chain(clusters[i+1:], reversed(processed[:i])):
            if not (maybe_liftable & set(c.ispace.dimensions)):
                break
            if c is candidate:
                continue
            if set(a.function for a in candidate.scope.accesses) & set(c.scope.writes):
                assert not candidate.atomics
                legal = False
                break

        if legal:
            # Contracted iteration and data spaces
            key = lambda d: d not in maybe_liftable
            ispace = candidate.ispace.project(key)
            dspace = candidate.dspace.project(key)
            # Now schedule at the right place
            lifted = Cluster(candidate.exprs, ispace, dspace)
            try:
                processed.insert(processed.index(c), lifted)
            except ValueError:
                # Still at the front
                processed.append(lifted)
        else:
            processed.append(candidate)

    return processed


def fuse(clusters):
    """
    Fuse PartialClusters to create "fatter" PartialClusters, that is
    PartialClusters containing more expressions. The topological ordering
    is honored.
    """
    # Clusters will be modified in-place in case of fusion
    clusters = [PartialCluster(*c.args) for c in clusters]

    processed = []
    for c in clusters:
        fused = False
        for candidate in reversed(list(processed)):
            # Guarded clusters cannot be grouped together
            if c.guards:
                break

            # Retrieve data dependences
            scope = Scope(exprs=candidate.exprs + c.exprs)

            # Collect anti-dependences preventing grouping
            anti = scope.d_anti.carried() - scope.d_anti.increment

            # Collect flow-dependences breaking the search
            flow = scope.d_flow - (scope.d_flow.inplace() + scope.d_flow.increment)

            # Can we group `c` with `candidate`?
            if candidate.ispace.is_compatible(c.ispace) and not anti \
                    and not candidate.guards:
                # Yes, `c` can be grouped with `candidate`: the iteration spaces
                # are compatible, there are no anti-dependences and no conditionals
                candidate.merge(c)
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
            elif any(i.is_Array for i in c.scope.writes):
                # Optimization: since the Cluster contains local Arrays (i.e.,
                # write-once/read-once Arrays), it might be convenient *not* to
                # attempt fusion with earlier Clusters: local Arrays often
                # store aliasing expressions (captured by the DSE), which are
                # prone to cross-loop blocking by the DLE, and we do not want to
                # break this optimization opportunity
                break
            elif set(candidate.guards) & set(c.dimensions):
                # Like above, we can't attempt fusion with earlier Clusters.
                # This time because there are intervening conditionals along
                # one or more of the shared iteration dimensions
                break
        # Fallback
        if not fused:
            processed.append(c)

    return processed


def guard(clusters):
    """
    Return a ClusterGroup containing a new PartialCluster for each conditional
    expression encountered in ``clusters``.
    """
    processed = []
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

    return processed


def reschedule(clusters):
    """
    Given a topologically-ordered sequence of Clusters, produce a new
    topologically-ordered sequence in which the following optimizations
    have been applied:

        * Fusion
        * Lifting

    Notes
    -----
    This function relies on advanced data dependency analysis tools based upon
    classic Lamport theory.
    """
    clusters = lift(clusters)
    clusters = fuse(clusters)
    return clusters


def clusterize(exprs):
    """Turn a sequence of LoweredEqs into a sequence of Clusters."""
    clusters = []

    # Wrap each LoweredEq in `exprs` within a PartialCluster. The PartialCluster's
    # iteration direction is enforced based on the iteration direction of the
    # surrounding LoweredEqs
    flowmap = detect_flow_directions(exprs)
    for e in exprs:
        directions, _ = force_directions(flowmap, lambda d: e.ispace.directions.get(d))
        ispace = IterationSpace(e.ispace.intervals, e.ispace.sub_iterators, directions)

        clusters.append(PartialCluster(e, ispace, e.dspace))

    # Cluster fusion
    clusters = fuse(clusters)

    # Introduce conditional PartialClusters
    clusters = guard(clusters)

    return ClusterGroup(clusters)

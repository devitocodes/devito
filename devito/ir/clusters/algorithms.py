from itertools import chain

import sympy

from devito.ir.support import (Scope, IterationSpace, detect_flow_directions,
                               force_directions)
from devito.ir.clusters.cluster import PartialCluster, ClusterGroup
from devito.symbolics import CondEq

__all__ = ['clusterize', 'reschedule']


def lift(clusters, cache=None):
    """
    PartialClusters may be invariant in one or more Dimensions; lifting
    will remove such Dimensions while honoring topological ordering as
    well as data dependences.
    """
    cache = cache or ScopeCache()

#    processed = []
#    for candidate in clusters:
#        # It is a lifting candidate if it redundantly computes values,
#        # that is if the iteration space contains more Dimensions than
#        # strictly needed (as there are no binding data dependences)
#        maybe_liftable = (set(candidate.ispace.dimensions) - 
#                          set().union(*[e.free_symbols for e in candidate.exprs]))
#        # Would the data dependences be honored?
#        for c in clusters[clusters.index(candidate) + 1:]:
#            if maybe_liftable & set(c.ispace.dimensions):
#                scope = cache.putdefault(candidate, c)
#            else:
#                break
#            processed.append(c)
#        else:
#            processed.append(c)

    processed = clusters
    return processed


def fuse(clusters, cache=None):
    """
    Fuse PartialClusters to create "fatter" PartialClusters, that is
    PartialClusters containing more expressions. The topological ordering
    as well as data dependences are honored.
    """
    cache = cache or ScopeCache()

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
            scope = cache.putdefault(candidate, c)

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
            elif any(i.is_Array for i in c.accesses):
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
    cache = ScopeCache()

    clusters = lift(clusters, cache)

    clusters = fuse(clusters, cache)

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


class ScopeCache(dict):

    def putdefault(self, *clusters):
        key = tuple(clusters)
        if key in self:
            return self[key]
        else:
            scope = Scope(exprs=chain(*[c.exprs for c in clusters]))
            self[key] = scope
            return scope

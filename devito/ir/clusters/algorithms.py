from itertools import chain

import sympy

from devito.ir.support import (Scope, IterationSpace, detect_flow_directions,
                               force_directions)
from devito.ir.clusters.cluster import PartialCluster, Cluster, ClusterGroup
from devito.symbolics import CondEq
from devito.types import Dimension

__all__ = ['clusterize', 'schedule']


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

    # Apply optimizations
    clusters = schedule(clusters)

    # Introduce conditional PartialClusters
    clusters = guard(clusters)

    return ClusterGroup(clusters)


def schedule(clusters):
    """
    Produce a topologically-ordered sequence of Clusters. The anti-dependences between
    two Clusters, which enforce a strict ordering in the execution of operations, are
    encoded as metadata in the `atomics` attribute.

    In the process, the following optimizations are applied:

        * Fusion
        * Lifting

    Notes
    -----
    This function relies on advanced data dependency analysis tools based upon classic
    Lamport theory.
    """

    # Build a dependence DAG. An edge from a Cluster A to a Cluster B means that B needs
    # to be executed before A due to a "direct dependence". A direct dependence is either
    # an anti-dependence or a dimension-independent flow dependence


    return clusters


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

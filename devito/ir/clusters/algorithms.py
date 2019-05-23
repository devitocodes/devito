from itertools import chain, groupby

from cached_property import cached_property
import sympy

from devito.ir.support import (Scope, IterationSpace, detect_flow_directions,
                               force_directions)
from devito.ir.clusters.cluster import PartialCluster, Cluster, ClusterGroup
from devito.symbolics import CondEq
from devito.types import Dimension
from devito.tools import DAG, DefaultOrderedDict, as_tuple

__all__ = ['clusterize', 'schedule']


def clusterize(exprs):
    """
    Turn a sequence of LoweredEqs into a sequence of Clusters.
    """
    # Initialization
    clusters = [PartialCluster(e, e.ispace, e.dspace) for e in exprs]

    # Topological sorting and optimizations
    clusters = schedule(clusters)

    # Introduce conditional PartialClusters
    clusters = guard(clusters)

    return ClusterGroup(clusters)


def schedule(clusters):
    """
    Produce a topologically-ordered sequence of Clusters while introducing the
    following optimizations:

        * Fusion
        * Lifting

    Notes
    -----
    This function relies on advanced data dependency analysis tools based upon classic
    Lamport theory.
    """
    csequences = [ClusterSequence(c, c.itintervals) for c in clusters]

    scheduler = Scheduler()
    csequences = scheduler.process(csequences)

    clusters = ClusterSequence.concatenate(*csequences)

    return clusters


class Scheduler(object):

    """
    A scheduler for ClusterSequences.
    """

    def __init__(self, callbacks=None):
        self.callabacks = as_tuple(callbacks)

    def _process(self, csequences, level):
        if all(level > len(cs.itintervals) for cs in csequences):
            # TODO: apply callbacks here
            return csequences
        else:
            key = lambda i: i.itintervals[:level]
            processed = []
            for k, g in groupby(csequences, key=key):
                if level > len(k):
                    continue
                else:
                    processed.extend(self._process(list(g), level + 1))
            return processed

    def process(self, csequences):
        return self._process(csequences, 1)


class ClusterSequence(tuple):

    """
    A totally-ordered sequence of Clusters.
    """

    def __new__(cls, items, itintervals):
        obj = super(ClusterSequence, cls).__new__(cls, as_tuple(items))
        obj._itintervals = itintervals
        return obj

    def __repr__(self):
        return "ClusterSequence([%s])" % ','.join('%s' % c for c in self)

    @classmethod
    def concatenate(cls, *csequences):
        return list(chain(*csequences))

    @cached_property
    def exprs(self):
        return list(chain(c.exprs) for c in self)

    @cached_property
    def scope(self):
        return Scope(self.exprs)

    @cached_property
    def itintervals(self):
        """The prefix IterationIntervals common to all Clusters in self."""
        return self._itintervals


def build_dag(csequences, dimensions):
    """
    A DAG capturing dependences between ClusterSequences along a given set
    of Dimensions.

    Examples
    --------
    When do we need to sequentialize two ClusterSequence `cs0` and `cs1` ?
    Assume `cs0` and `cs1` have same iteration space and `dimensions = {i}`.

    1) cs0 := b[i, j] = ...
       cs1 := ... = ... b[i+1, j] ...
       Anti-dependence in `i`, so `cs1` must go after `cs0`

    2) cs0 := b[i, j] = ...
       cs1 := ... = ... b[i-1, j+1] ...
       Flow-dependence in `i`, so `cs1` can safely go before or after `cs0`
    """
    dag = DAG(nodes=csequences)
    for i, cs0 in enumerate(csequences):
        for cs1 in csequences[i+1:]:
            scope = Scope(exprs=cs0.exprs + cs1.exprs)

            local_deps = list(chain(cs0.scope.d_all, cs1.scope.d_all))
            if any(dep.cause - set(dimensions) for dep in scope.d_all):
                dag.add_edge(cs0, cs1)
                break
    return dag


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

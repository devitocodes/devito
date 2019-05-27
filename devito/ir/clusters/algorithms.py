from collections import Counter
from itertools import chain, groupby

from cached_property import cached_property
import sympy

from devito.ir.support import (Any, DataSpace, IterationSpace, Scope,
                               detect_flow_directions, force_directions)
from devito.ir.clusters.cluster import Cluster, ClusterGroup
from devito.symbolics import CondEq
from devito.types import Dimension
from devito.tools import DAG, DefaultOrderedDict, as_tuple, flatten

__all__ = ['clusterize', 'schedule']


def clusterize(exprs):
    """
    Turn a sequence of LoweredEqs into a sequence of Clusters.
    """
    # Initialization
    clusters = [Cluster(e, e.ispace, e.dspace) for e in exprs]

    # Enforce iteration direction
    clusters = Queue(enforce).process(clusters)

    # Apply optimizations
    clusters = schedule(clusters)

    # Introduce conditional Clusters
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

    csequences = Queue(toposort, fuse, lift).process(csequences)
    clusters = ClusterSequence.concatenate(*csequences)

    return clusters


def enforce(clusters, prefix):
    """
    Replace, where necessary, `Any` IterationDirections with either `Forward` or
    `Backward`. This is accomplished by analyzing dependences across Clusters.
    """
    # Don't waste time building a flow map if there's a single Cluster
    if len(clusters) == 1:
        return clusters

    mapper = detect_flow_directions(flatten(c.exprs for c in clusters))

    processed = []
    for c in clusters:
        directions, _ = force_directions(mapper, lambda d: c.ispace.directions.get(d))
        ispace = IterationSpace(c.ispace.intervals, c.ispace.sub_iterators, directions)
        processed.append(Cluster(c.exprs, ispace, c.dspace))

    return processed


def toposort(csequences, prefix):
    """
    A new heuristic-based topological ordering for some ClusterSequences. The
    heuristic attempts to maximize Cluster fusion by bringing together Clusters
    with compatible IterationSpace.
    """
    # Are there any ClusterSequences that could potentially be fused? If not,
    # don't waste time computing a new topological ordering
    counter = Counter(cs.itintervals for cs in csequences)
    if not any(v > 1 for it, v in counter.most_common()):
        return csequences

    # Similarly, if all ClusterSequences have the same exact prefix, no need
    # to topologically resort
    if len(counter.most_common()) == 1:
        return csequences

    dag = build_dag(csequences, prefix)

    # TODO: choose_element-based toposort

    return csequences


def fuse(csequences, prefix):
    """
    Fuse ClusterSequences with compatible IterationSpace.
    """
    processed = []
    for k, g in groupby(csequences, key=lambda cs: cs.itintervals):
        maybe_fusible = list(g)
        clusters = ClusterSequence.concatenate(*maybe_fusible)
        if len(clusters) == 1 or\
                any(c.guards or c.itintervals != prefix for c in clusters):
            processed.extend(maybe_fusible)
        else:
            fused = Cluster.from_clusters(*clusters)
            processed.append(ClusterSequence(fused, fused.itintervals))
    return processed


def lift(csequences, prefix):
    # TODO: implement me
    # no-op ATM
    return csequences


class Queue(object):

    """
    A special queue to process objects in nested IterationSpaces based on
    a divide-and-conquer algorithm.

    Parameters
    ----------
    callbacks : list of callables, optional
        Routines executed upon conquer.
    """

    def __init__(self, *callbacks):
        self.callbacks = as_tuple(callbacks)

    def _process(self, elements, level, prefix=None):
        # Divide part
        processed = []
        for pfx, g in groupby(elements, key=lambda i: i.itintervals[:level]):
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Recursion
                processed.extend(self._process(list(g), level + 1, pfx))
        # Conquer part (execute callbacks)
        for f in self.callbacks:
            processed = f(processed, prefix)
        return processed

    def process(self, elements):
        return self._process(elements, 1)


class ClusterSequence(tuple):

    """
    A totally-ordered sequence of Clusters.
    """

    def __new__(cls, items, itintervals):
        obj = super(ClusterSequence, cls).__new__(cls, flatten(as_tuple(items)))
        obj._itintervals = itintervals
        return obj

    def __repr__(self):
        return "ClusterSequence([%s])" % ','.join('%s' % c for c in self)

    @classmethod
    def concatenate(cls, *csequences):
        return list(chain(*csequences))

    @cached_property
    def exprs(self):
        return flatten(c.exprs for c in self)

    @cached_property
    def itintervals(self):
        """The prefix IterationIntervals common to all Clusters in self."""
        return self._itintervals


def build_dag(csequences, prefix):
    """
    A DAG capturing dependences between ClusterSequences.

    The section of IterationSpace common to all ClusterSequences is described
    via ``prefix``, a tuple of IterationIntervals.

    Examples
    --------
    When do we need to sequentialize two ClusterSequence `cs0` and `cs1` ?

    Assume `prefix=[i]`

    1) cs0 := b[i, j] = ...
       cs1 := ... = ... b[i+1, j] ...
       Anti-dependence in `i`, so `cs1` must go after `cs0`

    2) cs0 := b[i, j] = ...
       cs1 := ... = ... b[i-1, j+1] ...
       Flow-dependence in `i`, so `cs1` can safely go before or after `cs0`

    Now assume `prefix=[]`

    3) cs0 := b[i, j] = ...
       cs1 := ... = ... b[i, j-1] ...
       Flow-dependence in `j`, but the `i` IterationInterval is different (e.g.,
       `i*[0,0]` for `cs0` and `i*[-1, 1]` for `cs1`), so `cs1` must go after `cs0`.
    """
    prefix = set(as_tuple(prefix))
    dag = DAG(nodes=csequences)
    for i, cs0 in enumerate(csequences):
        for cs1 in csequences[i+1:]:
            scope = Scope(exprs=cs0.exprs + cs1.exprs)
            if any(dep.cause - prefix for dep in scope.d_all):
                dag.add_edge(cs0, cs1)
                break
    return dag


def guard(clusters):
    """
    Split Clusters containing conditional expressions into separate Clusters.
    """
    processed = []
    for c in clusters:
        free = []
        for e in c.exprs:
            if e.conditionals:
                # Expressions that need no guarding are kept in a separate Cluster
                if free:
                    processed.append(Cluster(free, c.ispace, c.dspace, c.atomics))
                    free = []
                # Create a guarded Cluster
                guards = {}
                for d in e.conditionals:
                    condition = guards.setdefault(d.parent, [])
                    condition.append(d.condition or CondEq(d.parent % d.factor, 0))
                guards = {k: sympy.And(*v, evaluate=False) for k, v in guards.items()}
                processed.append(Cluster(e, c.ispace, c.dspace, c.atomics, guards))
            else:
                free.append(e)
        # Leftover
        if free:
            processed.append(Cluster(free, c.ispace, c.dspace, c.atomics))

    return processed

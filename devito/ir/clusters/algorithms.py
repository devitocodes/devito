from collections import Counter
from itertools import chain, groupby

from cached_property import cached_property
import sympy

from devito.ir.support import Scope, DataSpace, IterationSpace, force_directions
from devito.ir.clusters.cluster import PartialCluster, Cluster, ClusterGroup
from devito.symbolics import CondEq
from devito.types import Dimension
from devito.tools import DAG, DefaultOrderedDict, as_tuple, flatten

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

    scheduler = Scheduler(toposort, fuse)
    csequences = scheduler.process(csequences)

    clusters = ClusterSequence.concatenate(csequences)

    return clusters


def enforce_directions(csequences):
    pass


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


def lift(csequences):
    # TODO: implement me
    # no-op ATM
    return csequences


class Scheduler(object):

    """
    A scheduler for ClusterSequences.

    The scheduler adopts a divide-and-conquer algorithm. [... TODO ...]
    """

    def __init__(self, *callbacks):
        self.callbacks = as_tuple(callbacks)

    def _process(self, csequences, level, prefix=None):
        if all(level > len(cs.itintervals) for cs in csequences):
            for f in self.callbacks:
                csequences = f(csequences, prefix)
            return ClusterSequence(csequences, prefix)
        else:
            processed = []
            for k, g in groupby(csequences, key=lambda i: i.itintervals[:level]):
                if level > len(k):
                    continue
                else:
                    processed.extend(self._process(list(g), level + 1, k))
            return processed

    def process(self, csequences):
        return self._process(csequences, 1)


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
        return list(chain(c.exprs) for c in self)

    @cached_property
    def scope(self):
        return Scope(self.exprs)

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
    prefix = set(prefix)
    dag = DAG(nodes=csequences)
    for i, cs0 in enumerate(csequences):
        for cs1 in csequences[i+1:]:
            scope = Scope(exprs=cs0.exprs + cs1.exprs)

            local_deps = list(chain(cs0.scope.d_all, cs1.scope.d_all))
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

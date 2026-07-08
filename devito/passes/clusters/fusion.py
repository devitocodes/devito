from collections import Counter, defaultdict
from functools import cached_property
from itertools import groupby

from devito.finite_differences import IndexDerivative
from devito.ir.clusters import Cluster, ClusterGroup, Queue
from devito.ir.support import (
    InitArray, PrefetchUpdate, ReleaseLock, Scope, SyncArray, WaitLock, WithLock
)
from devito.symbolics import search
from devito.tools import (
    DAG, CacheInstances, as_tuple, flatten, frozendict, memoized_func, timed_pass
)

__all__ = ['fuse']


# No hazard: fusion may proceed.
NO_HAZARD = None
# Ordering hazard: preserve program order and forbid fusion.
EDGE = 'edge'
# Prefix anti-dependence: break the execution flow across the pair.
BREAK = 'break'


@memoized_func(scope='build')
def _fusion_hazards(scope0, scope1, prefix):
    """
    Classify the dependence hazard that would arise from fusing two scopes.
    """
    scope = Scope.from_scopes(scope0, scope1)
    if scope is None:
        return NO_HAZARD

    anti = False
    for i in scope.d_anti_gen():
        if i.cause & prefix:
            return BREAK
        anti = True

    if anti:
        return EDGE

    for i in scope.d_flow_gen():
        if not (i.cause & prefix):
            return EDGE

    for _ in scope.d_output_gen():
        return EDGE

    return NO_HAZARD


class Keys(CacheInstances):

    """
    Provide different kind of keys for Clusters (ClusterGroups) to be used in
    topological reordering and fusion.
    """

    def __init__(self, c, fuse_tasks):
        self.c = c
        self.fuse_tasks = fuse_tasks

    @property
    def first(self):
        return self.c[0]

    @property
    def last(self):
        return self.c[-1]

    @cached_property
    def itintervals(self):
        return self.c.ispace.itintervals

    @cached_property
    def guards(self):
        return self.c.guards if any(self.c.guards) else None

    @cached_property
    def syncs(self):
        mapper = defaultdict(set)
        for d, v in self.c.syncs.items():
            for s in v:
                if isinstance(s, PrefetchUpdate):
                    continue
                elif isinstance(s, WaitLock) and not self.fuse_tasks:
                    # NOTE: A mix of Clusters w/ and w/o WaitLocks can safely
                    # be fused, as in the worst case scenario the WaitLocks
                    # get "hoisted" above the first Cluster in the sequence
                    continue
                elif isinstance(s, (InitArray, SyncArray, WaitLock, ReleaseLock)):
                    mapper[d].add(type(s))
                elif isinstance(s, WithLock) and self.fuse_tasks:
                    # NOTE: Different WithLocks aren't fused unless the user
                    # explicitly asks for it
                    mapper[d].add(type(s))
                else:
                    mapper[d].add(s)
            if d in mapper:
                mapper[d] = frozenset(mapper[d])
        return frozendict(mapper)

    @cached_property
    def strict(self):
        return (self.itintervals, self.guards, self.syncs)

    @cached_property
    def weak(self):
        c = self.c

        # Clusters representing HaloTouches should get merged, if possible
        weak = [c.is_halo_touch]

        # If there are writes to thread-shared object, make it part of the key.
        # This will promote fusion of non-adjacent Clusters writing to (some
        # form of) shared memory, which in turn will minimize the number of
        # necessary barriers. Same story for reads from thread-shared objects
        weak.extend([
            any(f._mem_shared for f in c.scope.writes),
            any(f._mem_shared for f in c.scope.reads)
        ])
        weak.append(c.properties.is_core_init())

        # Prefetchable Clusters should get merged, if possible
        weak.append(c.is_glb_load_to_mem_shared)

        # Promoting adjacency of IndexDerivatives will maximize their reuse
        weak.append(any(search(c.exprs, IndexDerivative)))

        # Promote adjacency of Clusters with same guard
        weak.append(c.guards)

        return tuple(weak)

    @cached_property
    def full(self):
        return self.strict + self.weak


class Fusion(Queue):

    """
    Fuse Clusters with compatible IterationSpace.
    """

    _q_guards_in_key = True
    _q_syncs_in_key = True

    def __init__(self, toposort, options=None):
        options = options or {}

        self.toposort = toposort
        self.fuse_tasks = options.get('fuse-tasks', False)

        super().__init__()

    def process(self, clusters):
        cgroups = [ClusterGroup(c, c.ispace) for c in clusters]
        cgroups = self._process_fdta(cgroups, 1)
        clusters = ClusterGroup.concatenate(*cgroups)
        return clusters

    def callback(self, cgroups, prefix):
        # Toposort to maximize fusion
        if self.toposort:
            clusters = self._toposort(cgroups, prefix)
            if self.toposort == 'nofuse':
                return [clusters]
        else:
            clusters = ClusterGroup(cgroups)

        # Fusion
        key = lambda c: self._key(c).full
        processed = []
        for _, group in groupby(clusters, key=key):
            g = list(group)

            for maybe_fusible in self._apply_heuristics(g):
                try:
                    # Perform fusion
                    processed.append(Cluster.from_clusters(*maybe_fusible))
                except ValueError:
                    # We end up here if, for example, some Clusters have same
                    # iteration Dimensions but different (partial) orderings
                    processed.extend(maybe_fusible)

        # Maximize effectiveness of topo-sorting at next stage by only
        # grouping together Clusters characterized by data dependencies
        if self.toposort and prefix:
            dag = self._build_dag(processed, prefix)
            mapper = dag.connected_components(enumerated=True)
            groups = groupby(processed, key=mapper.get)
            return [ClusterGroup(tuple(g), prefix) for _, g in groups]
        else:
            return [ClusterGroup(processed, prefix)]

    def _key(self, c):
        return Keys(c, self.fuse_tasks)

    def _apply_heuristics(self, clusters):
        # We know at this point that `clusters` are potentially fusible since
        # they have same `_key`, but should we actually fuse them? In most cases
        # yes, but there are exceptions...

        # 1) Consider the following scenario with three Clusters:
        #  c0[no syncs]
        #  c1[WaitLock]
        #  c2[no syncs]
        # Then we return two groups [[c0], [c1, c2]] rather than a single group
        # [[c0, c1, c2]] because this way c0 can be computed without having to
        # wait on a lock for a longer period
        processed = []

        group = []
        flag = False  # True -> need to dump before creating a new group

        def dump():
            processed.append(tuple(group))
            group[:] = []

        for c in clusters:
            if any(isinstance(i, WaitLock) for i in flatten(c.syncs.values())):
                if flag:
                    dump()
                    flag = False
            else:
                flag = True
            group.append(c)
        dump()

        # 2) Don't group HaloTouch's
        groups, processed = processed, []
        for group in groups:
            for flag, minigroup in groupby(group, key=lambda c: c.is_wild):
                if flag:
                    processed.extend([(c,) for c in minigroup])
                else:
                    processed.append(tuple(minigroup))

        return processed

    def _toposort(self, cgroups, prefix):
        # If not enough ClusterGroups to do anything meaningful, don't waste time
        if len(cgroups) <= 2:
            return ClusterGroup(cgroups, prefix)

        # Are there any ClusterGroups that could potentially be topologically
        # reordered? If not, do not waste time
        counter = Counter(self._key(cg).strict for cg in cgroups)
        if len(counter.most_common()) == 1 or \
           not any(v > 1 for it, v in counter.most_common()):
            return ClusterGroup(cgroups, prefix)

        dag = self._build_dag(cgroups, prefix)

        def choose_element(queue, scheduled):
            if not scheduled or len(queue) == 1:
                return queue.pop()

            k = self._key(scheduled[-1])
            m = {i: self._key(i) for i in queue}

            # First of all, ensure we preserve the integrity of the current scope
            candidates = [i for i in queue if k.last.ispace == m[i].first.ispace]

            compatible = [i for i in candidates if k.last.guards == m[i].first.guards]
            candidates = compatible or candidates

            # If the current scope is over, we maximize fusion
            fusible = [i for i in queue if k.itintervals == m[i].itintervals]
            candidates = candidates or fusible

            compatible = [i for i in candidates if k.syncs == m[i].syncs]
            candidates = compatible or candidates

            # Process the `weak` part of the key
            for i in range(len(k.weak), -1, -1):
                choosable = [e for e in candidates if m[e].weak[:i] == k.weak[:i]]
                try:
                    # Ensure stability
                    e = min(choosable, key=lambda i: cgroups.index(i))
                except ValueError:
                    continue
                queue.remove(e)
                return e

            # Fallback
            e = min(queue, key=lambda i: cgroups.index(i))
            queue.remove(e)
            return e

        return ClusterGroup(dag.topological_sort(choose_element), prefix)

    def _build_dag(self, cgroups, prefix):
        """
        A DAG representing the data dependences across the ClusterGroups within
        a given scope.
        """
        prefix = frozenset(i.dim for i in as_tuple(prefix))

        dag = DAG(nodes=cgroups)
        for n, cg0 in enumerate(cgroups):
            # Track whether there is any fence between `cg0` and the current `cg1`.
            fenced = cg0.scope.has_barrier

            for n1, cg1 in enumerate(cgroups[n+1:], start=n+1):
                fenced = fenced or cg1.scope.has_barrier

                hazard = _fusion_hazards(cg0.scope, cg1.scope, prefix)
                if not (hazard or fenced):
                    continue

                # Anti-dependences along `prefix` break the execution flow
                # (intuitively, "the loop nests are to be kept separated")
                # * All ClusterGroups between `cg0` and `cg1` must precede `cg1`
                # * All ClusterGroups after `cg1` cannot precede `cg1`
                if hazard == BREAK:
                    for cg2 in cgroups[n:n1]:
                        dag.add_edge(cg2, cg1)
                    for cg2 in cgroups[n1+1:]:
                        dag.add_edge(cg1, cg2)
                    break
                elif fenced or hazard == EDGE:
                    # Any anti- and iaw-dependences impose that `cg1` follows `cg0`
                    # and forbid any sort of fusion. Fences have the same effect
                    dag.add_edge(cg0, cg1)

        return dag


@timed_pass()
def fuse(clusters, toposort=False, options=None):
    """
    Clusters fusion.

    If `toposort=True`, then the Clusters are reordered to maximize the likelihood
    of fusion; the new ordering is computed such that all data dependencies are
    honored.

    If `toposort='maximal'`, then `toposort` is performed, iteratively, multiple
    times to actually maximize Clusters fusion. Hence, this is more aggressive than
    `toposort=True`.
    """
    if toposort != 'maximal':
        return Fusion(toposort, options).process(clusters)

    nxt = clusters
    while True:
        nxt = fuse(clusters, toposort='nofuse', options=options)
        if all(c0 is c1 for c0, c1 in zip(clusters, nxt, strict=True)):
            break
        clusters = nxt
    clusters = fuse(clusters, toposort=False, options=options)

    return clusters

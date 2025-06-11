from collections import Counter, defaultdict
from itertools import groupby, product

from devito.finite_differences import IndexDerivative
from devito.ir.clusters import Cluster, ClusterGroup, Queue, cluster_pass
from devito.ir.support import (SEQUENTIAL, SEPARABLE, Scope, ReleaseLock, WaitLock,
                               WithLock, InitArray, SyncArray, PrefetchUpdate)
from devito.passes.clusters.utils import in_critical_region
from devito.symbolics import pow_to_mul, search
from devito.tools import DAG, Stamp, as_tuple, flatten, frozendict, timed_pass
from devito.types import Hyperplane

__all__ = ['Lift', 'fuse', 'optimize_pows', 'fission', 'optimize_hyperplanes']


class Lift(Queue):

    """
    Remove invariant Dimensions from Clusters to avoid redundant computation.

    Notes
    -----
    This is analogous to the compiler transformation known as
    "loop-invariant code motion".
    """

    @timed_pass(name='lift')
    def process(self, elements):
        return super().process(elements)

    def callback(self, clusters, prefix):
        if not prefix:
            # No iteration space to be lifted from
            return clusters
        dim = prefix[-1].dim

        hope_invariant = dim._defines
        outer = set().union(*[i.dim._defines for i in prefix[:-1]])

        lifted = []
        processed = []
        for n, c in enumerate(clusters):
            # Storage-related dependences, such as those induced by reduction
            # increments, prevent lifting
            if any(dep.is_storage_related(dim) for dep in c.scope.d_all_gen()):
                processed.append(c)
                continue

            # Synchronization prevents lifting
            if any(c.syncs.get(d) for d in dim._defines) or \
               in_critical_region(c, clusters):
                processed.append(c)
                continue

            # Is `c` a real candidate -- is there at least one invariant Dimension?
            if any(d._defines & hope_invariant for d in c.used_dimensions):
                processed.append(c)
                continue

            impacted = set(processed) | set(clusters[n+1:])

            # None of the Functions appearing in a lifted Cluster can be written to
            if any(c.functions & set(i.scope.writes) for i in impacted):
                processed.append(c)
                continue

            # All of the inner Dimensions must appear in the write-to region
            # otherwise we would violate data dependencies. Consider
            #
            # 1)                 2)                          3)
            # for i              for i                       for i
            #   for x              for x                       for x
            #     r = f(a[x])        for y                       for y
            #                          r[x] = f(a[x, y])           r[x, y] = f(a[x, y])
            #
            # In 1) and 2) lifting is infeasible; in 3) the statement can
            # be lifted outside the `i` loop as `r`'s write-to region contains
            # both `x` and `y`
            xed = {d._defines for d in c.used_dimensions if d not in outer}
            if not all(i & set(w.dimensions) for i, w in product(xed, c.scope.writes)):
                processed.append(c)
                continue

            # The contracted iteration and data spaces
            key = lambda d: d not in hope_invariant
            ispace = c.ispace.project(key)

            # Optimization: if not lifting from the innermost Dimension, we can
            # safely reset the `ispace` to expose potential fusion opportunities
            try:
                if c.ispace.innermost.dim not in hope_invariant:
                    ispace = ispace.reset()
            except IndexError:
                pass

            properties = c.properties.filter(key)

            # Lifted scalar clusters cannot be guarded
            # as they would not be in the scope of the guarded clusters
            # unless the guard is for an outer dimension
            if c.is_scalar and not (prefix[:-1] and c.guards):
                guards = {}
            else:
                guards = c.guards

            lifted.append(c.rebuild(ispace=ispace, properties=properties, guards=guards))

        return lifted + processed


class Fusion(Queue):

    """
    Fuse Clusters with compatible IterationSpace.
    """

    _q_guards_in_key = True
    _q_syncs_in_key = True

    def __init__(self, toposort, options=None):
        options = options or {}

        self.toposort = toposort
        self.fusetasks = options.get('fuse-tasks', False)

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
        processed = []
        for k, group in groupby(clusters, key=self._key):
            g = list(group)

            for maybe_fusible in self._apply_heuristics(g):
                if len(maybe_fusible) == 1:
                    processed.extend(maybe_fusible)
                else:
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

    class Key(tuple):

        """
        A fusion Key for a Cluster (ClusterGroup) is a hashable tuple such that
        two Clusters (ClusterGroups) are topo-fusible if and only if their Key is
        identical.

        A Key contains elements that can logically be split into two groups -- the
        `strict` and the `weak` components of the Key. Two Clusters (ClusterGroups)
        having same `strict` but different `weak` parts are, by definition, not
        fusible; however, since at least their `strict` parts match, they can at
        least be topologically reordered.
        """

        def __new__(cls, itintervals, guards, syncs, weak):
            strict = [itintervals, guards, syncs]
            obj = super().__new__(cls, strict + weak)

            obj.itintervals = itintervals
            obj.guards = guards
            obj.syncs = syncs

            obj.strict = tuple(strict)
            obj.weak = tuple(weak)

            return obj

    def _key(self, c):
        itintervals = frozenset(c.ispace.itintervals)
        guards = c.guards if any(c.guards) else None

        # We allow fusing Clusters/ClusterGroups even in presence of WaitLocks and
        # WithLocks, but not with any other SyncOps
        mapper = defaultdict(set)
        for d, v in c.syncs.items():
            for s in v:
                if isinstance(s, PrefetchUpdate):
                    continue
                elif isinstance(s, WaitLock) and not self.fusetasks:
                    # NOTE: A mix of Clusters w/ and w/o WaitLocks can safely
                    # be fused, as in the worst case scenario the WaitLocks
                    # get "hoisted" above the first Cluster in the sequence
                    continue
                elif isinstance(s, (InitArray, SyncArray, WaitLock, ReleaseLock)):
                    mapper[d].add(type(s))
                elif isinstance(s, WithLock) and self.fusetasks:
                    # NOTE: Different WithLocks aren't fused unless the user
                    # explicitly asks for it
                    mapper[d].add(type(s))
                else:
                    mapper[d].add(s)
            if d in mapper:
                mapper[d] = frozenset(mapper[d])
        syncs = frozendict(mapper)

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
        weak.append(c.properties.is_prefetchable_shm())

        # Promoting adjacency of IndexDerivatives will maximize their reuse
        weak.append(any(search(c.exprs, IndexDerivative)))

        # Promote adjacency of Clusters with same guard
        weak.append(c.guards)

        key = self.Key(itintervals, guards, syncs, weak)

        return key

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
        # Are there any ClusterGroups that could potentially be topologically
        # reordered? If not, do not waste time
        counter = Counter(self._key(cg).strict for cg in cgroups)
        if len(counter.most_common()) == 1 or \
           not any(v > 1 for it, v in counter.most_common()):
            return ClusterGroup(cgroups, prefix)

        dag = self._build_dag(cgroups, prefix)

        def choose_element(queue, scheduled):
            if not scheduled:
                return queue.pop()

            k = self._key(scheduled[-1])
            m = {i: self._key(i) for i in queue}

            # Process the `strict` part of the key
            candidates = [i for i in queue if m[i].itintervals == k.itintervals]

            compatible = [i for i in candidates if m[i].guards == k.guards]
            candidates = compatible or candidates

            compatible = [i for i in candidates if m[i].syncs == k.syncs]
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
        prefix = {i.dim for i in as_tuple(prefix)}

        dag = DAG(nodes=cgroups)
        for n, cg0 in enumerate(cgroups):

            def is_cross(source, sink):
                # True if a cross-ClusterGroup dependence, False otherwise
                t0 = source.timestamp
                t1 = sink.timestamp
                v = len(cg0.exprs)
                return t0 < v <= t1 or t1 < v <= t0

            for n1, cg1 in enumerate(cgroups[n+1:], start=n+1):

                # A Scope to compute all cross-ClusterGroup anti-dependences
                scope = Scope(exprs=cg0.exprs + cg1.exprs, rules=is_cross)

                # Anti-dependences along `prefix` break the execution flow
                # (intuitively, "the loop nests are to be kept separated")
                # * All ClusterGroups between `cg0` and `cg1` must precede `cg1`
                # * All ClusterGroups after `cg1` cannot precede `cg1`
                if any(i.cause & prefix for i in scope.d_anti_gen()):
                    for cg2 in cgroups[n:cgroups.index(cg1)]:
                        dag.add_edge(cg2, cg1)
                    for cg2 in cgroups[cgroups.index(cg1)+1:]:
                        dag.add_edge(cg1, cg2)
                    break

                # Any anti- and iaw-dependences impose that `cg1` follows `cg0`
                # and forbid any sort of fusion. Fences have the same effect
                elif (any(scope.d_anti_gen()) or
                      any(i.is_iaw for i in scope.d_output_gen()) or
                      any(c.is_fence for c in flatten(cgroups[n:n1+1]))):
                    dag.add_edge(cg0, cg1)

                # Any flow-dependences along an inner Dimension (i.e., a Dimension
                # that doesn't appear in `prefix`) impose that `cg1` follows `cg0`
                elif any(not (i.cause and i.cause & prefix)
                         for i in scope.d_flow_gen()):
                    dag.add_edge(cg0, cg1)

                # Clearly, output dependences must be honored
                elif any(scope.d_output_gen()):
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
        if all(c0 is c1 for c0, c1 in zip(clusters, nxt)):
            break
        clusters = nxt
    clusters = fuse(clusters, toposort=False, options=options)

    return clusters


@cluster_pass(mode='all')
def optimize_pows(cluster, *args):
    """
    Convert integer powers into Muls, such as ``a**2 => a*a``.
    """
    return cluster.rebuild(exprs=[pow_to_mul(e) for e in cluster.exprs])


class Fission(Queue):

    """
    Implement Clusters fission. For more info refer to fission.__doc__.
    """

    def callback(self, clusters, prefix):
        if not prefix or len(clusters) == 1:
            return clusters

        d = prefix[-1].dim

        # Do not waste time if definitely illegal
        if any(SEQUENTIAL in c.properties[d] for c in clusters):
            return clusters

        # Do not waste time if definitely nothing to do
        if all(len(prefix) == len(c.ispace) for c in clusters):
            return clusters

        # Analyze and abort if fissioning would break a dependence
        scope = Scope(flatten(c.exprs for c in clusters))
        if any(d._defines & dep.cause or dep.is_reduce(d) for dep in scope.d_all_gen()):
            return clusters

        processed = []
        for (it, guards), g in groupby(clusters, key=lambda c: self._key(c, prefix)):
            group = list(g)

            try:
                test0 = any(SEQUENTIAL in c.properties[it.dim] for c in group)
            except AttributeError:
                # `it` is None because `c`'s IterationSpace has no `d` Dimension,
                # hence `key = (it, guards) = (None, guards)`
                test0 = True

            if test0 or guards:
                # Heuristic: no gain from fissioning if unable to ultimately
                # increase the number of collapsable iteration spaces, hence give up
                processed.extend(group)
            else:
                stamp = Stamp()
                for c in group:
                    ispace = c.ispace.lift(d, stamp)
                    processed.append(c.rebuild(ispace=ispace))

        return processed

    def _key(self, c, prefix):
        try:
            index = len(prefix)
            dims = tuple(i.dim for i in prefix)

            it = c.ispace[index]
            guards = frozendict({d: v for d, v in c.guards.items() if d in dims})

            return (it, guards)
        except IndexError:
            return (None, c.guards)


@timed_pass()
def fission(clusters):
    """
    Clusters fission.

    Currently performed in the following cases:

        * Trade off data locality for parallelism, e.g.

          .. code-block::

            for x              for x
              for y1             for y1
                ..                 ..
              for y2     -->   for x
                ..               for y2
                                   ..
    """
    return Fission().process(clusters)


@timed_pass()
def optimize_hyperplanes(clusters):
    """
    At the moment this is just a dummy no-op pass that we only use
    for testing purposes.
    """
    for c in clusters:
        for k, v in c.properties.items():
            if isinstance(k, Hyperplane) and SEPARABLE in v:
                raise NotImplementedError

    return clusters

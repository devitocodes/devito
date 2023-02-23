from collections import Counter, defaultdict
from itertools import groupby, product

from devito.ir.clusters import Cluster, ClusterGroup, Queue, cluster_pass
from devito.ir.support import (SEQUENTIAL, SEPARABLE, Scope, ReleaseLock,
                               WaitLock, WithLock, FetchUpdate, PrefetchUpdate)
from devito.symbolics import pow_to_mul
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
        return super(Lift, self).process(elements)

    def callback(self, clusters, prefix):
        if not prefix:
            # No iteration space to be lifted from
            return clusters

        hope_invariant = prefix[-1].dim._defines
        outer = set().union(*[i.dim._defines for i in prefix[:-1]])

        lifted = []
        processed = []
        for n, c in enumerate(clusters):
            # Increments prevent lifting
            if c.has_increments:
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
            # In 1) and 2) lifting is infeasible; in 3) the statement can be lifted
            # outside the `i` loop as `r`'s write-to region contains both `x` and `y`
            xed = {d._defines for d in c.used_dimensions if d not in outer}
            if not all(i & set(w.dimensions) for i, w in product(xed, c.scope.writes)):
                processed.append(c)
                continue

            # The contracted iteration and data spaces
            key = lambda d: d not in hope_invariant
            ispace = c.ispace.project(key)

            # Optimization: if not lifting from the innermost Dimension, we can
            # safely reset the `ispace` to expose potential fusion opportunities
            if c.ispace[-1].dim not in hope_invariant:
                ispace = ispace.reset()

            properties = {d: v for d, v in c.properties.items() if key(d)}

            lifted.append(c.rebuild(ispace=ispace, properties=properties))

        return lifted + processed


class Fusion(Queue):

    """
    Fuse Clusters with compatible IterationSpace.
    """

    _q_guards_in_key = True

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

    def _key(self, c):
        # Two Clusters/ClusterGroups are fusion candidates if their key is identical

        key = (frozenset(c.ispace.itintervals),)

        # If there are writes to thread-shared object, make it part of the key.
        # This will promote fusion of non-adjacent Clusters writing to (some form of)
        # shared memory, which in turn will minimize the number of necessary barriers
        key += (any(f._mem_shared for f in c.scope.writes),)
        # Same story for reads from thread-shared objects
        key += (any(f._mem_shared for f in c.scope.reads),)

        key += (c.guards if any(c.guards) else None,)

        # We allow fusing Clusters/ClusterGroups even in presence of WaitLocks and
        # WithLocks, but not with any other SyncOps
        if isinstance(c, Cluster):
            syncs = (c.syncs,)
        else:
            syncs = c.syncs
        for i in syncs:
            mapper = defaultdict(set)
            for k, v in i.items():
                for s in v:
                    if isinstance(s, (FetchUpdate, PrefetchUpdate)) or \
                       (not self.fusetasks and isinstance(s, WaitLock)):
                        # NOTE: A mix of Clusters w/ and w/o WaitLocks can safely
                        # be fused, as in the worst case scenario the WaitLocks
                        # get "hoisted" above the first Cluster in the sequence
                        continue
                    elif (isinstance(s, (WaitLock, ReleaseLock)) or
                          (self.fusetasks and isinstance(s, WithLock))):
                        mapper[k].add(type(s))
                    else:
                        mapper[k].add(s)
                mapper[k] = frozenset(mapper[k])
            if any(mapper.values()):
                mapper = frozendict(mapper)
                key += (mapper,)

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

        return processed

    def _toposort(self, cgroups, prefix):
        # Are there any ClusterGroups that could potentially be fused? If
        # not, do not waste time computing a new topological ordering
        counter = Counter(self._key(cg) for cg in cgroups)
        if not any(v > 1 for it, v in counter.most_common()):
            return ClusterGroup(cgroups, prefix)

        # Similarly, if all ClusterGroups have the same exact prefix and
        # use the same form of synchronization (if any at all), no need to
        # attempt a topological sorting
        if len(counter.most_common()) == 1:
            return ClusterGroup(cgroups, prefix)

        dag = self._build_dag(cgroups, prefix)

        def choose_element(queue, scheduled):
            if not scheduled:
                return queue.pop()

            # Heuristic: let `k0` be the key of the last scheduled node; then out of
            # the possible schedulable nodes we pick the one with key `k1` such that
            # `max_i : k0[:i] == k1[:i]` (i.e., the one with "the most similar key")
            key = self._key(scheduled[-1])
            for i in reversed(range(len(key) + 1)):
                candidates = [e for e in queue if self._key(e)[:i] == key[:i]]
                try:
                    # Ensure stability
                    e = min(candidates, key=lambda i: cgroups.index(i))
                except ValueError:
                    continue
                queue.remove(e)
                return e

            assert False

        return ClusterGroup(dag.topological_sort(choose_element), prefix)

    def _build_dag(self, cgroups, prefix, peeking=False):
        """
        A DAG representing the data dependences across the ClusterGroups within
        a given scope.
        """
        prefix = {i.dim for i in as_tuple(prefix)}

        dag = DAG(nodes=cgroups)
        for n, cg0 in enumerate(cgroups):

            def is_cross(dep):
                # True if a cross-ClusterGroup dependence, False otherwise
                t0 = dep.source.timestamp
                t1 = dep.sink.timestamp
                v = len(cg0.exprs)
                return t0 < v <= t1 or t1 < v <= t0

            for cg1 in cgroups[n+1:]:
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
                # while not being its immediate successor (unless it already is),
                # to avoid they are fused together (thus breaking the dependence)
                # TODO: the "not being its immediate successor" part *seems* to be
                # a work around to the fact that any two Clusters characterized
                # by anti-dependence should have been given a different stamp,
                # and same for guarded Clusters, but that is not the case (yet)
                elif any(scope.d_anti_gen()) or\
                        any(i.is_iaw for i in scope.d_output_gen()):
                    dag.add_edge(cg0, cg1)
                    index = cgroups.index(cg1) - 1
                    if index > n and self._key(cg0) == self._key(cg1):
                        dag.add_edge(cg0, cgroups[index])
                        dag.add_edge(cgroups[index], cg1)

                # Any flow-dependences along an inner Dimension (i.e., a Dimension
                # that doesn't appear in `prefix`) impose that `cg1` follows `cg0`
                elif any(not (i.cause and i.cause & prefix) for i in scope.d_flow_gen()):
                    dag.add_edge(cg0, cg1)

                # Clearly, output dependences must be honored
                elif any(scope.d_output_gen()):
                    dag.add_edge(cg0, cg1)

            if peeking and dag.edges:
                return dag

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

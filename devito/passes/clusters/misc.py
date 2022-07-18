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
            ispace = c.ispace.project(key).reset()

            properties = {d: v for d, v in c.properties.items() if key(d)}

            lifted.append(c.rebuild(ispace=ispace, properties=properties))

        return lifted + processed


class Fusion(Queue):

    """
    Fuse Clusters with compatible IterationSpace.
    """

    def __init__(self, toposort, options=None):
        options = options or {}

        self.toposort = toposort
        self.fusetasks = options.get('fuse-tasks', False)

        super().__init__()

    def _make_key_hook(self, cgroup, level):
        assert level > 0
        assert len(cgroup.guards) == 1
        return (tuple(cgroup.guards[0].get(i.dim) for i in cgroup.itintervals[:level-1]),)

    def process(self, clusters):
        cgroups = [ClusterGroup(c, c.itintervals) for c in clusters]
        cgroups = self._process_fdta(cgroups, 1)
        clusters = ClusterGroup.concatenate(*cgroups)
        return clusters

    def callback(self, cgroups, prefix):
        # Toposort to maximize fusion
        if self.toposort:
            clusters = self._toposort(cgroups, prefix)
        else:
            clusters = ClusterGroup(cgroups)

        # Fusion
        processed = []
        for k, g in groupby(clusters, key=self._key):
            maybe_fusible = list(g)

            if len(maybe_fusible) == 1:
                processed.extend(maybe_fusible)
            else:
                try:
                    # Perform fusion
                    fused = Cluster.from_clusters(*maybe_fusible)
                    processed.append(fused)
                except ValueError:
                    # We end up here if, for example, some Clusters have same
                    # iteration Dimensions but different (partial) orderings
                    processed.extend(maybe_fusible)

        return [ClusterGroup(processed, prefix)]

    def _key(self, c):
        # Two Clusters/ClusterGroups are fusion candidates if their key is identical

        key = (frozenset(c.itintervals), c.guards)

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
                    if isinstance(s, (FetchUpdate, PrefetchUpdate)):
                        continue
                    elif (isinstance(s, (WaitLock, ReleaseLock)) or
                          (self.fusetasks and isinstance(s, WithLock))):
                        mapper[k].add(type(s))
                    else:
                        mapper[k].add(s)
                mapper[k] = frozenset(mapper[k])
            mapper = frozendict(mapper)
            key += (mapper,)

        return key

    def _toposort(self, cgroups, prefix):
        # Are there any ClusterGroups that could potentially be fused? If
        # not, do not waste time computing a new topological ordering
        counter = Counter(self._key(cg) for cg in cgroups)
        if not any(v > 1 for it, v in counter.most_common()):
            return ClusterGroup(cgroups)

        # Similarly, if all ClusterGroups have the same exact prefix and
        # use the same form of synchronization (if any at all), no need to
        # attempt a topological sorting
        if len(counter.most_common()) == 1:
            return ClusterGroup(cgroups)

        dag = self._build_dag(cgroups, prefix)

        def choose_element(queue, scheduled):
            # Heuristic: let `k0` be the key of the last scheduled node; then out of
            # the possible schedulable nodes we pick the one with key `k1` such that
            # `max_i : k0[:i] == k1[:i]` (i.e., the one with "the most similar key")
            if not scheduled:
                return queue.pop()
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

        return ClusterGroup(dag.topological_sort(choose_element))

    def _build_dag(self, cgroups, prefix):
        """
        A DAG representing the data dependences across the ClusterGroups within
        a given scope.
        """
        prefix = {i.dim for i in as_tuple(prefix)}

        dag = DAG(nodes=cgroups)
        for n, cg0 in enumerate(cgroups):
            for cg1 in cgroups[n+1:]:
                # A Scope to compute all cross-ClusterGroup anti-dependences
                rule = lambda i: i.is_cross
                scope = Scope(exprs=cg0.exprs + cg1.exprs, rules=rule)

                # Optimization: we exploit the following property:
                # no prefix => (edge <=> at least one (any) dependence)
                # to jump out of this potentially expensive loop as quickly as possible
                if not prefix and any(scope.d_all_gen()):
                    dag.add_edge(cg0, cg1)

                # Anti-dependences along `prefix` break the execution flow
                # (intuitively, "the loop nests are to be kept separated")
                # * All ClusterGroups between `cg0` and `cg1` must precede `cg1`
                # * All ClusterGroups after `cg1` cannot precede `cg1`
                elif any(i.cause & prefix for i in scope.d_anti_gen()):
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

        return dag


@timed_pass()
def fuse(clusters, toposort=False, options=None):
    """
    Clusters fusion.

    If ``toposort=True``, then the Clusters are reordered to maximize the likelihood
    of fusion; the new ordering is computed such that all data dependencies are honored.
    """
    return Fusion(toposort, options).process(clusters)


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
        if all(len(prefix) == len(c.itintervals) for c in clusters):
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

            it = c.itintervals[index]
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

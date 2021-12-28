from collections import OrderedDict, defaultdict

import numpy as np
from sympy import And

from devito.ir import Forward, GuardBound, GuardBoundNext, Queue, Vector, SEQUENTIAL
from devito.symbolics import uxreplace
from devito.tools import (DefaultOrderedDict, frozendict, is_integer,
                          indices_to_sections, timed_pass)
from devito.types import (CustomDimension, Lock, WaitLock, WithLock, FetchUpdate,
                          FetchPrefetch, PrefetchUpdate, WaitPrefetch, Delete,
                          normalize_syncs)

__all__ = ['Tasker', 'Streaming']


class Asynchronous(Queue):

    def __init__(self, key):
        assert callable(key)
        self.key = key
        super().__init__()


class Tasker(Asynchronous):

    """
    Create asynchronous Clusters, or "tasks".

    Parameters
    ----------
    key : callable, optional
        A Cluster `c` becomes an asynchronous task only if `key(c)` returns True

    Notes
    -----
    From an implementation viewpoint, an asynchronous Cluster is a Cluster
    with attached suitable SyncOps, such as WaitLock, WithLock, etc.
    """

    @timed_pass(name='tasker')
    def process(self, clusters):
        return super().process(clusters)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        if not all(SEQUENTIAL in c.properties[d] for c in clusters):
            return clusters

        locks = {}
        waits = defaultdict(list)
        tasks = defaultdict(list)
        for c0 in clusters:
            if not self.key(c0):
                # Not a candidate asynchronous task
                continue

            # Prevent future writes to interfere with a task by waiting on a lock
            may_require_lock = c0.scope.reads

            # We can ignore scalars as they're passed by value
            may_require_lock = {f for f in may_require_lock if f.is_AbstractFunction}

            # Sort for deterministic code generation
            may_require_lock = sorted(may_require_lock, key=lambda i: i.name)

            protected = defaultdict(set)
            for c1 in clusters:
                offset = int(clusters.index(c1) <= clusters.index(c0))

                for f in may_require_lock:
                    try:
                        writes = c1.scope.writes[f]
                    except KeyError:
                        # No read-write dependency, ignore
                        continue

                    try:
                        if all(w.aindices[d].is_Stepping for w in writes) or \
                           all(w.aindices[d].is_Modulo for w in writes):
                            size = f.shape_allocated[d]
                            assert is_integer(size)
                            ld = CustomDimension(name='ld', symbolic_size=size, parent=d)
                        elif all(w[d] == 0 for w in writes):
                            # Special case, degenerates to scalar lock
                            raise KeyError
                        else:
                            # Functions over non-stepping Dimensions need no lock
                            continue
                    except KeyError:
                        # Would degenerate to a scalar, but we rather use a lock
                        # of size 1 for simplicity
                        ld = CustomDimension(name='ld', symbolic_size=1)
                    lock = locks.setdefault(f, Lock(
                        name='lock%d' % len(locks), dimensions=ld, target=f,
                        initvalue=np.full(ld.symbolic_size, 2, dtype=np.int32)
                    ))

                    for w in writes:
                        try:
                            index = w[d]
                            logical_index = index + offset
                        except TypeError:
                            assert ld.symbolic_size == 1
                            index = 0
                            logical_index = 0

                        if logical_index in protected[f]:
                            continue

                        waits[c1].append(WaitLock(lock[index]))
                        protected[f].add(logical_index)

            # Taskify `c0`
            for f in protected:
                lock = locks[f]

                indices = sorted({r[d] for r in c0.scope.reads[f]})
                if indices == [None]:
                    # `lock` is protecting a Function which isn't defined over `d`
                    # E.g., `d=time` and the protected function is `a(x, y)`
                    assert lock.size == 1
                    indices = [0]

                tasks[c0].extend(WithLock(lock[i]) for i in indices)

        processed = []
        for c in clusters:
            if waits[c] or tasks[c]:
                processed.append(c.rebuild(syncs={d: waits[c] + tasks[c]}))
            else:
                processed.append(c)

        return processed


class Streaming(Asynchronous):

    """
    Tag Clusters with SyncOps to stream Functions in and out the device memory.

    Parameters
    ----------
    key : callable, optional
        Return the Functions that need to be streamed in a given Cluster.
    """

    @timed_pass(name='streaming')
    def process(self, clusters):
        return super().process(clusters)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        it = prefix[-1]
        d = it.dim

        actions = defaultdict(Actions)

        # Case 1
        if d.is_Custom and is_integer(it.size):
            for c in clusters:
                candidates = self.key(c)
                if candidates:
                    if is_memcpy(c):
                        # Case 1A (special case, leading to more efficient streaming)
                        self._actions_from_init(c, prefix, actions)
                    else:
                        # Case 1B (actually, we expect to never end up here)
                        raise NotImplementedError

        # Case 2
        elif all(SEQUENTIAL in c.properties[d] for c in clusters):
            mapper = OrderedDict()
            for c in clusters:
                candidates = self.key(c)
                if candidates:
                    mapper[c] = is_memcpy(c)

            # Case 2A (special case, leading to more efficient streaming)
            if all(mapper.values()):
                for c in mapper:
                    self._actions_from_update_memcpy(c, clusters, prefix, actions)

            # Case 2B
            elif mapper:
                self._actions_from_unstructured(clusters, prefix, actions)

        # Perform the necessary actions; this will ultimately attach SyncOps to Clusters
        processed = []
        for c in clusters:
            v = actions[c]

            if v.drop:
                assert not v.syncs
                continue
            elif v.syncs:
                processed.append(c.rebuild(syncs=normalize_syncs(c.syncs, v.syncs)))
            else:
                processed.append(c)

            if v.insert:
                processed.extend(v.insert)

        return processed

    def _actions_from_init(self, cluster, prefix, actions):
        return actions_from_init(cluster, prefix, actions)

    def _actions_from_update_memcpy(self, cluster, clusters, prefix, actions):
        return actions_from_update_memcpy(cluster, clusters, prefix, actions)

    def _actions_from_unstructured(self, clusters, prefix, actions):
        return actions_from_unstructured(clusters, self.key, prefix, actions)


# Utilities


class Actions(object):

    def __init__(self, drop=False, syncs=None, insert=None):
        self.drop = drop
        self.syncs = syncs or defaultdict(list)
        self.insert = insert or []


def is_memcpy(cluster):
    """
    True if `cluster` emulates a memcpy and the target object is a mapped
    Array, False otherwise.
    """
    return (len(cluster.exprs) == 1 and
            cluster.exprs[0].lhs.function.is_Array and
            cluster.exprs[0].lhs.function._mem_mapped and
            cluster.exprs[0].rhs.is_Indexed)


def actions_from_init(cluster, prefix, actions):
    it = prefix[-1]
    d = it.dim
    try:
        pd = prefix[-2].dim
    except IndexError:
        pd = None

    # Prepare the data to instantiate a FetchUpdate SyncOp
    e = cluster.exprs[0]

    size = d.symbolic_size

    function = e.rhs.function
    fetch = e.rhs.indices[d]
    ifetch = fetch.subs(d, d.symbolic_min)
    fcond = None

    pfetch = None
    pcond = None

    target = e.lhs.function
    tstore = 0

    # Sanity checks
    assert is_integer(size)

    actions[cluster].syncs[pd].append(FetchUpdate(
        d, size,
        function, fetch, ifetch, fcond,
        pfetch, pcond,
        target, tstore
    ))


def actions_from_update_memcpy(cluster, clusters, prefix, actions):
    it = prefix[-1]
    d = it.dim
    direction = it.direction

    # Prepare the data to instantiate a PrefetchUpdate SyncOp
    e = cluster.exprs[0]

    size = 1

    function = e.rhs.function
    fetch = e.rhs.indices[d]
    ifetch = fetch.subs(d, d.symbolic_min)
    fcond = None

    if direction is Forward:
        pfetch = fetch + 1
    else:
        pfetch = fetch - 1
    pcond = None

    target = e.lhs.function
    tstore0 = e.lhs.indices[d]

    # If fetching into e.g., `ub[sb1]`, we'll need to prefetch into e.g. `ub[sb0]`
    if is_integer(tstore0):
        tstore = tstore0
    else:
        assert tstore0.is_Modulo
        subiters = [md for md in cluster.sub_iterators[d] if md.parent is tstore0.parent]
        osubiters = sorted(subiters, key=lambda i: Vector(i.offset))
        n = osubiters.index(tstore0)
        if direction is Forward:
            tstore = osubiters[(n + 1) % len(osubiters)]
        else:
            tstore = osubiters[(n - 1) % len(osubiters)]

    # Turn `cluster` into a prefetch Cluster
    expr = uxreplace(e, {tstore0: tstore, fetch: pfetch})

    guards = {d: And(
        cluster.guards.get(d, True),
        GuardBoundNext(function.indices[d], direction),
    )}

    syncs = {d: [PrefetchUpdate(
        d, size,
        function, fetch, ifetch, fcond,
        pfetch, pcond,
        target, tstore
    )]}

    pcluster = cluster.rebuild(exprs=expr, guards=guards, syncs=syncs)

    # Since we're turning `e` into a prefetch, we need to:
    # 1) attach a WaitPrefetch SyncOp to the first Cluster accessing `target`
    # 2) insert the prefetch Cluster right after the last Cluster accessing `target`
    # 3) drop the original Cluster performing a memcpy-based fetch
    n = clusters.index(cluster)
    first = None
    last = None
    for c in clusters[n+1:]:
        if target in c.scope.reads:
            if first is None:
                first = c
            last = c
    assert first is not None
    assert last is not None
    actions[first].syncs[d].append(WaitPrefetch(
        d, size,
        function, fetch, ifetch, fcond,
        pfetch, pcond,
        target, tstore
    ))
    actions[last].insert.append(pcluster)
    actions[cluster].drop = True

    return last, pcluster


def actions_from_unstructured(clusters, key, prefix, actions):
    it = prefix[-1]
    d = it.dim
    direction = it.direction

    # Locate the streamable Functions
    first_seen = {}
    last_seen = {}
    for c in clusters:
        candidates = key(c)
        if not candidates:
            continue
        for i in c.scope.accesses:
            f = i.function
            if f in candidates:
                k = (f, i[d])
                first_seen.setdefault(k, c)
                last_seen[k] = c
    if not first_seen:
        return clusters

    callbacks = [(frozendict(first_seen), FetchPrefetch),
                 (frozendict(last_seen), Delete)]

    # Create and map SyncOps to Clusters
    for seen, callback in callbacks:
        mapper = defaultdict(lambda: DefaultOrderedDict(list))
        for (f, v), c in seen.items():
            mapper[c][f].append(v)

        for c, m in mapper.items():
            for f, v in m.items():
                for fetch, s in indices_to_sections(v):
                    if direction is Forward:
                        ifetch = fetch.subs(d, d.symbolic_min)
                        pfetch = fetch + 1
                    else:
                        ifetch = fetch.subs(d, d.symbolic_max)
                        pfetch = fetch - 1

                    fcond = GuardBound(d.symbolic_min, d.symbolic_max)
                    pcond = GuardBoundNext(f.indices[d], direction)

                    syncop = callback(d, s, f, fetch, ifetch, fcond, pfetch, pcond)
                    actions[c].syncs[d].append(syncop)

from collections import OrderedDict, defaultdict

from sympy import Ge, Le, Mod, Mul, true

from devito.exceptions import InvalidOperator
from devito.ir.clusters import Queue
from devito.ir.support import Forward, SEQUENTIAL, Vector
from devito.tools import (DefaultOrderedDict, frozendict, is_integer,
                          indices_to_sections, timed_pass)
from devito.types import (CustomDimension, Lock, WaitLock, WithLock, Fetch,
                          FetchPrefetch, FetchMemcpy, PrefetchMemcpy, WaitPrefetch,
                          Delete, normalize_syncs)

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
            may_require_lock = set(c0.scope.reads)

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
                    lock = locks.setdefault(f, Lock(name='lock%d' % len(locks),
                                                    dimensions=ld, target=f))

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

        sync_ops = defaultdict(lambda: defaultdict(list))

        # Case 1
        if d.is_Custom and is_integer(it.size):
            for c in clusters:
                candidates = self.key(c)
                if candidates:
                    # Case 1A (special case, leading to more efficient streaming)
                    if is_memcpy(c):
                        update_syncops_from_init(c, prefix, sync_ops, memcpy=True)
                    # Case 1B
                    elif is_memcpy_like(c):
                        update_syncops_from_init(c, prefix, sync_ops)
                    else:
                        raise NotImplementedError

        # Case 2
        elif all(SEQUENTIAL in c.properties[d] for c in clusters):
            mapper = OrderedDict()
            for c in clusters:
                candidates = self.key(c)
                if candidates:
                    if is_memcpy(c):
                        mapper[c] = update_syncops_from_update_memcpy
                    else:
                        mapper[c] = None

            # Case 2A (special case, leading to more efficient streaming)
            if all(i is update_syncops_from_update_memcpy for i in mapper.values()):
                for c in mapper:
                    update_syncops_from_update_memcpy(c, clusters, prefix, sync_ops)

            # Case 2B
            elif mapper:
                update_syncops_from_unstructured(clusters, self.key, prefix, sync_ops)

        # Attach SyncOps to Clusters
        processed = []
        for c in clusters:
            syncs = sync_ops.get(c)
            if syncs == 'drop':
                continue
            elif syncs:
                processed.append(c.rebuild(syncs=normalize_syncs(c.syncs, syncs)))
            else:
                processed.append(c)

        return processed


# Utilities


def is_memcpy(cluster):
    """
    True if `cluster` emulates a memcpy and the target object is a default-allocated
    Array, False otherwise.
    """
    return (len(cluster.exprs) == 1 and
            cluster.exprs[0].lhs.function.is_Array and
            cluster.exprs[0].lhs.function._mem_default and
            cluster.exprs[0].rhs.is_Indexed)


def is_memcpy_like(cluster):
    """
    True if `cluster` emulates a memcpy, False otherwise.
    """
    return (len(cluster.exprs) == 1 and
            cluster.exprs[0].rhs.is_Indexed)


def update_syncops_from_init(cluster, prefix, sync_ops, memcpy=False):
    it = prefix[-1]
    d = it.dim
    direction = it.direction
    try:
        pd = prefix[-2].dim
    except IndexError:
        pd = None

    # Prepare the data to instantiate a FetchMemcpy SyncOp
    e = cluster.exprs[0]

    size = d.symbolic_size

    function = e.rhs.function
    fetch = e.rhs.indices[d]
    ifetch = fetch.subs(d, d.symbolic_min)
    if direction is Forward:
        fcond = make_cond(cluster.guards.get(d), d, direction, d.symbolic_min)
    else:
        fcond = make_cond(cluster.guards.get(d), d, direction, d.symbolic_max)

    pfetch = None
    pcond = None

    target = e.lhs.function
    tstore = e.lhs.indices[d].subs(d, d.symbolic_min)

    # Sanity checks
    assert is_integer(size)

    if memcpy:
        sync_ops[cluster][pd].append(FetchMemcpy(
            d, size,
            function, fetch, ifetch, fcond,
            pfetch, pcond,
            target, tstore
        ))
    else:
        sync_ops[cluster][pd].append(Fetch(
            d, size,
            function, ifetch, ifetch, fcond,
            pfetch, pcond,
            target, tstore
        ))
        sync_ops[cluster][pd].append(Delete(
            d, size,
            function, ifetch, ifetch, fcond,  # Note: `ifetch` twice, that's deliberate
            pfetch, pcond,
            target, tstore
        ))


def update_syncops_from_update_memcpy(cluster, clusters, prefix, sync_ops):
    it = prefix[-1]
    d = it.dim
    direction = it.direction

    # Prepare the data to instantiate a FetchMemcpy SyncOp
    e = cluster.exprs[0]

    size = 1

    function = e.rhs.function
    fetch = e.rhs.indices[d]
    ifetch = fetch.subs(d, d.symbolic_min)
    if direction is Forward:
        fcond = make_cond(cluster.guards.get(d), d, direction, d.symbolic_min)
    else:
        fcond = make_cond(cluster.guards.get(d), d, direction, d.symbolic_max)

    if direction is Forward:
        pfetch = fetch + 1
        pcond = make_cond(cluster.guards.get(d), d, direction, d + 1)
    else:
        pfetch = fetch - 1
        pcond = make_cond(cluster.guards.get(d), d, direction, d - 1)

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

    # Since we're turning `e` into a prefetch, we need to attach ...
    # ... a WaitPrefetch SyncOp to the first Cluster accessing the `target`
    # ... the actual PrefetchMemcpy to the last Cluster accessing the `target`
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
    sync_ops[first][d].append(WaitPrefetch(
        d, size,
        function, fetch, ifetch, fcond,
        pfetch, pcond,
        target, tstore
    ))
    sync_ops[last][d].append(PrefetchMemcpy(
        d, size,
        function, fetch, ifetch, fcond,
        pfetch, pcond,
        target, tstore
    ))
    sync_ops[cluster] = 'drop'


def update_syncops_from_unstructured(clusters, key, prefix, sync_ops):
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
                        fcond = make_cond(c.guards.get(d), d, direction, d.symbolic_min)
                        pfetch = fetch + 1
                        pcond = make_cond(c.guards.get(d), d, direction, d + 1)
                    else:
                        ifetch = fetch.subs(d, d.symbolic_max)
                        fcond = make_cond(c.guards.get(d), d, direction, d.symbolic_max)
                        pfetch = fetch - 1
                        pcond = make_cond(c.guards.get(d), d, direction, d - 1)

                    syncop = callback(d, s, f, fetch, ifetch, fcond, pfetch, pcond)
                    sync_ops[c][d].append(syncop)


def make_cond(rel, d, direction, iteration):
    """
    Create a symbolic condition which, once resolved at runtime, returns True
    if `iteration` is within the Dimension `d`'s min/max bounds, False otherwise.
    """
    if rel is None:
        if direction is Forward:
            cond = Le(iteration, d.symbolic_max)
        else:
            cond = Ge(iteration, d.symbolic_min)
    else:
        # Only case we know how to deal with, today, is the one induced
        # by a ConditionalDimension with structured condition (e.g. via `factor`)
        if not (rel.is_Equality and rel.rhs == 0 and isinstance(rel.lhs, Mod)):
            raise InvalidOperator("Unable to understand data streaming pattern")
        _, v = rel.lhs.args

        if direction is Forward:
            # The LHS rounds `s` up to the nearest multiple of `v`
            cond = Le(Mul(((iteration + v - 1) / v), v, evaluate=False), d.symbolic_max)
        else:
            # The LHS rounds `s` down to the nearest multiple of `v`
            cond = Ge(Mul((iteration / v), v, evaluate=False), d.symbolic_min)

    if cond is true:
        return None
    else:
        return cond

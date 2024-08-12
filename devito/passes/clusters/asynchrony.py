from collections import defaultdict

from sympy import true

from devito.ir import (Forward, Backward, GuardBoundNext, WaitLock, WithLock, SyncArray,
                       PrefetchUpdate, ReleaseLock, Queue, normalize_syncs)
from devito.passes.clusters.utils import in_critical_region, is_memcpy
from devito.symbolics import IntDiv, uxreplace
from devito.tools import OrderedSet, is_integer, timed_pass
from devito.types import CustomDimension, Lock

__all__ = ['tasking', 'memcpy_prefetch']


def async_trigger(c, dims):
    """
    Return the Dimension in `c`'s IterationSpace that triggers the
    asynchronous execution of `c`.
    """
    if not dims:
        return None
    else:
        key = lambda d: not d._defines.intersection(dims)
        ispace = c.ispace.project(key)
        return ispace.innermost.dim


def keys(key0):
    """
    Callbacks for `tasking` and `memcpy_prefetch` given a user-defined `key`.
    """

    def task_key(c):
        return async_trigger(c, key0(c.scope.writes))

    def memcpy_key(c):
        if task_key(c):
            # Writes would take precedence over reads
            return None
        else:
            return async_trigger(c, key0(c.scope.reads))

    return task_key, memcpy_key


@timed_pass(name='tasking')
def tasking(clusters, key0, sregistry):
    """
    Turn a Cluster `c` into an asynchronous task if `c` writes to a Function `f`
    such that `key0(f) = (d_i, ..., d_n)`, where `(d_i, ..., d_n)` are the
    task Dimensions.
    """
    key, _ = keys(key0)
    return Tasking(key, sregistry).process(clusters)


class Tasking(Queue):

    """
    Carry out the bulk of `tasking`.
    """

    def __init__(self, key0, sregistry):
        self.key0 = key0
        self.sregistry = sregistry

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        dim = prefix[-1].dim

        locks = {}
        syncs = defaultdict(lambda: defaultdict(OrderedSet))
        for c0 in clusters:
            d = self.key0(c0)
            if d is not dim:
                continue

            protected = self._schedule_waitlocks(c0, d, clusters, locks, syncs)
            self._schedule_withlocks(c0, d, protected, locks, syncs)

        processed = [c.rebuild(syncs={**c.syncs, **syncs[c]}) for c in clusters]

        return processed

    def _schedule_waitlocks(self, c0, d, clusters, locks, syncs):
        # Prevent future writes to interfere with a task by waiting on a lock
        may_require_lock = {f for f in c0.scope.reads if f.is_AbstractFunction}
        # Sort for deterministic code generation
        may_require_lock = sorted(may_require_lock, key=lambda i: i.name)

        protected = defaultdict(set)
        for c1 in clusters:
            offset = int(clusters.index(c1) <= clusters.index(c0))

            for target in may_require_lock:
                try:
                    writes = c1.scope.writes[target]
                except KeyError:
                    # No read-write dependency, ignore
                    continue

                try:
                    if all(w.aindices[d].is_Stepping for w in writes) or \
                       all(w.aindices[d].is_Modulo for w in writes):
                        sz = target.shape_allocated[d]
                        assert is_integer(sz)
                        ld = CustomDimension(name='ld', symbolic_size=sz, parent=d)
                    elif all(w[d] == 0 for w in writes):
                        # Special case, degenerates to scalar lock
                        raise KeyError
                    else:
                        # Functions over non-stepping Dimensions need no lock
                        continue
                except (AttributeError, KeyError):
                    # Would degenerate to a scalar, but we rather use a lock
                    # of size 1 for simplicity
                    ld = CustomDimension(name='ld', symbolic_size=1, parent=d)

                try:
                    lock = locks[target]
                except KeyError:
                    name = self.sregistry.make_name(prefix='lock')
                    lock = locks[target] = Lock(name=name, dimensions=ld)

                for w in writes:
                    try:
                        index = w[d]
                        logical_index = index + offset
                    except TypeError:
                        assert ld.symbolic_size == 1
                        index = 0
                        logical_index = 0

                    if logical_index in protected[target]:
                        continue

                    # Critical regions preempt WaitLocks
                    c2 = in_critical_region(c1, clusters) or c1

                    syncs[c2][d].add(WaitLock(lock[index], target))
                    protected[target].add(logical_index)

        return protected

    def _schedule_withlocks(self, c0, d, protected, locks, syncs):
        for target in protected:
            lock = locks[target]

            indices = sorted({r[d] for r in c0.scope.reads[target]})
            if indices == [None]:
                # `lock` is protecting a Function which isn't defined over `d`
                # E.g., `d=time` and the protected function is `a(x, y)`
                assert lock.size == 1
                indices = [0]

            if wraps_memcpy(c0):
                e = c0.exprs[0]
                function = e.lhs.function
                findex = e.lhs.indices[d]
            else:
                # Only for backwards compatibility (e.g., tasking w/o buffering)
                function = None
                findex = None

            for i in indices:
                syncs[c0][d].update([
                    ReleaseLock(lock[i], target),
                    WithLock(lock[i], target, i, function, findex, d)
                ])


@timed_pass(name='memcpy_prefetch')
def memcpy_prefetch(clusters, key0, sregistry):
    """
    A special form of `tasking` for optimized, asynchronous memcpy-like operations.
    Unlike `tasking`, `key0` is here applied to the reads of a given Cluster.
    """
    _, key = keys(key0)
    actions = defaultdict(Actions)

    for c in clusters:
        d = key(c)
        if d is None:
            continue

        if not wraps_memcpy(c):
            continue

        if c.properties.is_prefetchable(d._defines):
            _actions_from_update_memcpy(c, d, clusters, actions, sregistry)
        elif d.is_Custom and is_integer(c.ispace[d].size):
            _actions_from_init(c, d, actions)

    # Attach the computed Actions
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


def _actions_from_init(c, d, actions):
    e = c.exprs[0]
    function = e.rhs.function
    target = e.lhs.function

    tindex = e.lhs.indices[d]
    findex = e.rhs.indices[d]

    size = target.shape[d]
    assert is_integer(size)

    actions[c].syncs[d].append(
        SyncArray(None, target, tindex, function, findex, d, size)
    )


def _actions_from_update_memcpy(c, d, clusters, actions, sregistry):
    pd = d.root  # E.g., `vd -> time`
    direction = c.ispace[pd].direction

    e = c.exprs[0]
    function = e.rhs.function
    target = e.lhs.function

    fetch = e.rhs.indices[d]
    fshift = {Forward: 1, Backward: -1}.get(direction, 0)
    findex = fetch + fshift if fetch.find(IntDiv) else fetch._subs(pd, pd + fshift)

    # If fetching into e.g. `ub[t1]` we might need to prefetch into e.g. `ub[t0]`
    tindex0 = e.lhs.indices[d]
    if is_integer(tindex0) or isinstance(tindex0, IntDiv):
        tindex = tindex0
    else:
        assert tindex0.is_Modulo
        mapper = {(i.offset % i.modulo): i for i in c.sub_iterators[pd]}
        if direction is Forward:
            toffset = tindex0.offset + 1
        else:
            toffset = tindex0.offset - 1
        try:
            tindex = mapper[toffset % tindex0.modulo]
        except KeyError:
            # This can happen if e.g. the underlying buffer has size K (because
            # e.g. the user has sent `async_degree=K`), but the actual computation
            # only uses K-N indices (e.g., indices={t-1,t} and K=5})
            name = sregistry.make_name(prefix='t')
            tindex = tindex0._rebuild(name, offset=toffset, origin=None)

    # We need a lock to synchronize the copy-in
    name = sregistry.make_name(prefix='lock')
    ld = CustomDimension(name='ld', symbolic_size=1, parent=d)
    lock = Lock(name=name, dimensions=ld)
    handle = lock[0]

    # Turn `c` into a prefetch Cluster `pc`
    expr = uxreplace(e, {tindex0: tindex, fetch: findex})

    if tindex is not tindex0:
        ispace = c.ispace.augment({pd: tindex})
    else:
        ispace = c.ispace

    guard0 = c.guards.get(d, true)._subs(fetch, findex)
    guard1 = GuardBoundNext(function.indices[d], direction)
    guards = c.guards.impose(d, guard0 & guard1)

    syncs = {d: [
        ReleaseLock(handle, target),
        PrefetchUpdate(handle, target, tindex, function, findex, d, 1, e.rhs)
    ]}
    syncs = {**c.syncs, **syncs}

    pc = c.rebuild(exprs=expr, ispace=ispace, guards=guards, syncs=syncs)

    # Since we're turning `e` into a prefetch, we need to:
    # 1) attach a WaitLock SyncOp to the first Cluster accessing `target`
    # 2) insert the prefetch Cluster right after the last Cluster accessing `target`
    # 3) drop the original Cluster performing a memcpy-like fetch
    n = clusters.index(c)
    first = None
    last = None
    for c1 in clusters[n+1:]:
        if target in c1.scope.reads:
            if first is None:
                first = c1
            last = c1
    assert first is not None
    assert last is not None

    actions[first].syncs[d].append(WaitLock(handle, target))
    actions[last].insert.append(pc)
    actions[c].drop = True

    return last, pc


class Actions:

    def __init__(self, drop=False, syncs=None, insert=None):
        self.drop = drop
        self.syncs = syncs or defaultdict(list)
        self.insert = insert or []


def wraps_memcpy(cluster):
    return len(cluster.exprs) == 1 and is_memcpy(cluster.exprs[0])

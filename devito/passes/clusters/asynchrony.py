from collections import defaultdict

from sympy import Mod, Mul

from devito.exceptions import InvalidOperator
from devito.ir.clusters import Queue
from devito.ir.support import Forward, SEQUENTIAL
from devito.tools import (DefaultOrderedDict, frozendict, is_integer,
                          indices_to_sections, timed_pass)
from devito.types import (CustomDimension, Ge, Le, Lock, WaitLock, WithLock,
                          FetchWait, FetchWaitPrefetch, Delete, normalize_syncs)

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
    Tag Clusters with the FetchWait, FetchWaitPrefetch and Delete SyncOps to
    stream Functions in and out the process memory.

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
        direction = it.direction

        try:
            pd = prefix[-2].dim
        except IndexError:
            pd = None

        # What are the stream-able Dimensions?
        # 0) all sequential Dimensions
        # 1) all CustomDimensions of fixed (i.e. integer) size, which
        #    implies a bound on the amount of streamed data
        if all(SEQUENTIAL in c.properties[d] for c in clusters):
            make_fetch = lambda f, i, s, cb: FetchWaitPrefetch(f, d, direction, i, s, cb)
            make_delete = lambda f, i, s, cb: Delete(f, d, direction, i, s, cb)
            syncd = d
        elif d.is_Custom and is_integer(it.size):
            make_fetch = lambda f, i, s, cb: FetchWait(f, d, direction, i, it.size, cb)
            make_delete = lambda f, i, s, cb: Delete(f, d, direction, i, it.size, cb)
            syncd = pd
        else:
            return clusters

        first_seen = {}
        last_seen = {}
        for c in clusters:
            candidates = self.key(c)
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

        # Bind fetches and deletes to Clusters
        sync_ops = defaultdict(list)
        callbacks = [(frozendict(first_seen), make_fetch),
                     (frozendict(last_seen), make_delete)]
        for seen, callback in callbacks:
            mapper = defaultdict(lambda: DefaultOrderedDict(list))
            for (f, v), c in seen.items():
                mapper[c][f].append(v)
            for c, m in mapper.items():
                for f, v in m.items():
                    for i, s in indices_to_sections(v):
                        next_cbk = make_next_cbk(c.guards.get(d), d, direction)
                        sync_ops[c].append(callback(f, i, s, next_cbk))

        # Attach SyncOps to Clusters
        processed = []
        for c in clusters:
            v = sync_ops.get(c)
            if v is not None:
                processed.append(c.rebuild(syncs=normalize_syncs(c.syncs, {syncd: v})))
            else:
                processed.append(c)

        return processed


# Utilities

def make_next_cbk(rel, d, direction):
    """
    Create a callable that given a symbol returns a sympy.Relational usable to
    express, in symbolic form, whether the next fetch/prefetch will be executed.
    """
    if rel is None:
        if direction is Forward:
            return lambda s: Le(s, d.symbolic_max)
        else:
            return lambda s: Ge(s, d.symbolic_min)
    else:
        # Only case we know how to deal with, today, is the one induced
        # by a ConditionalDimension with structured condition (e.g. via `factor`)
        if not (rel.is_Equality and rel.rhs == 0 and isinstance(rel.lhs, Mod)):
            raise InvalidOperator("Unable to understand data streaming pattern")
        _, v = rel.lhs.args

        if direction is Forward:
            # The LHS rounds `s` up to the nearest multiple of `v`
            return lambda s: Le(Mul(((s + v - 1) / v), v, evaluate=False), d.symbolic_max)
        else:
            # The LHS rounds `s` down to the nearest multiple of `v`
            return lambda s: Ge(Mul((s / v), v, evaluate=False), d.symbolic_min)

from collections import OrderedDict, defaultdict
from itertools import chain

from cached_property import cached_property

from devito.ir.clusters import Queue, Cluster
from devito.ir.support import AFFINE, SEQUENTIAL, Backward, Scope
from devito.symbolics import uxreplace
from devito.tools import (DefaultOrderedDict, as_tuple, filter_ordered, flatten,
                          is_integer, timed_pass)
from devito.types import (Array, CustomDimension, ModuloDimension, Eq,
                          Lock, WaitLock, WithLock, WaitAndFetch, normalize_syncs)

__all__ = ['Tasker', 'Stream']


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
        A Cluster `c` becomes an asynchronous task only `key(f)` returns True
        for any of the Functions `f` in `c`.

    Notes
    -----
    From an implementation viewpoint, an asynchronous Cluster is a Cluster
    with attached suitable SyncOps, such as WaitLock, WithThread, etc.
    """

    @timed_pass(name='tasker')
    def process(self, clusters):
        return super().process(clusters)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        if not all(c.properties[d] >= {SEQUENTIAL, AFFINE} for c in clusters):
            return clusters

        locks = {}
        waits = defaultdict(list)
        tasks = defaultdict(list)
        for c0 in clusters:

            # Prevent future writes to interfere with a task by waiting on a lock
            may_require_lock = {i for i in c0.scope.reads
                                if any(self.key(w) for w in c0.scope.writes)}

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
                        if all(w.aindices[d].is_Stepping for w in writes):
                            size = f.shape_allocated[d]
                            assert is_integer(size)
                            ld = CustomDimension(name='ld', symbolic_size=size, parent=d)
                        else:
                            # Functions over non-stepping Dimensions need no lock
                            continue
                    except KeyError:
                        # Would degenerate to a scalar, but we rather use an Array
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


class Stream(Asynchronous):

    """
    Tag Clusters with the WaitAndFetch SyncOp to stream Functions in and out
    the process memory.

    Parameters
    ----------
    key : callable, optional
        A Function `f` in a Cluster `c` gets streamed only if `key(f)` returns True.
    """

    @timed_pass(name='stream')
    def process(self, clusters):
        return super().process(clusters)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim
        direction = prefix[-1].direction

        mapper = defaultdict(set)
        for c in clusters:
            if SEQUENTIAL not in c.properties[d]:
                continue

            for f, v in c.scope.reads.items():
                if not self.key(f):
                    continue
                if any(f in c1.scope.writes for c1 in clusters):
                    # Read-only Functions are the sole streaming candidates
                    continue

                mapper[f].update({i[d] for i in v})

        processed = []
        for c in clusters:

            syncs = []
            for f, v in list(mapper.items()):
                if f in c.scope.reads:
                    syncs.append(WaitAndFetch(f, d, direction, v))
                    mapper.pop(f)

            if syncs:
                processed.append(c.rebuild(syncs=normalize_syncs(c.syncs, {d: syncs})))
            else:
                processed.append(c)

        return processed

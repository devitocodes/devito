"""
Synchronization operations inside the IR.
"""

from collections import defaultdict

from devito.tools import Pickable, filter_ordered

__all__ = ['WaitLock', 'WithLock', 'FetchUpdate', 'FetchPrefetch',
           'PrefetchUpdate', 'WaitPrefetch', 'Delete', 'normalize_syncs']


class SyncOp(Pickable):

    is_SyncLock = False
    is_SyncData = False

    is_WaitLock = False
    is_WithLock = False
    is_Fetch = False

    def __eq__(self, other):
        return (type(self) == type(other) and
                all(i == j for i, j in zip(self.args, other.args)))

    def __hash__(self):
        return hash((type(self).__name__,) + self.args)

    @property
    def args(self):
        return ()


class SyncLock(SyncOp):

    is_SyncLock = True

    def __init__(self, handle):
        self.handle = handle

    def __repr__(self):
        return "%s<%s>" % (self.__class__.__name__, self.handle)

    __str__ = __repr__

    @property
    def args(self):
        return (self.handle,)

    @property
    def lock(self):
        return self.handle.function

    @property
    def target(self):
        return self.lock.target

    # Pickling support
    _pickle_args = ['handle']
    __reduce_ex__ = Pickable.__reduce_ex__


class SyncData(SyncOp):

    is_SyncData = True

    def __init__(self, dim, size, function, fetch, ifetch, fcond,
                 pfetch=None, pcond=None, target=None, tstore=None):

        # fetch -> the input Function fetch index, e.g. `time`
        # ifetch -> the input Function initialization index, e.g. `time_m`
        # pfetch -> the input Function prefetch index, e.g. `time+1`
        # tstore -> the target Function store index, e.g. `sb1`

        # fcond -> the input Function fetch condition, e.g. `time_m <= time_M`
        # pcond -> the input Function prefetch condition, e.g. `time + 1 <= time_M`

        self.dim = dim
        self.size = size
        self.function = function
        self.fetch = fetch
        self.ifetch = ifetch
        self.fcond = fcond
        self.pfetch = pfetch
        self.pcond = pcond
        self.target = target
        self.tstore = tstore

    def __repr__(self):
        return "%s<%s->%s:%s:%d>" % (self.__class__.__name__, self.function,
                                     self.target, self.dim, self.size)

    __str__ = __repr__

    @property
    def args(self):
        return (self.dim, self.size, self.function, self.fetch, self.ifetch,
                self.fcond, self.pfetch, self.pcond, self.target, self.tstore)

    @property
    def dimensions(self):
        return self.function.dimensions

    # Pickling support
    _pickle_args = ['dim', 'size', 'function', 'fetch', 'ifetch', 'fcond']
    _pickle_kwargs = ['pfetch', 'pcond', 'target', 'tstore']
    __reduce_ex__ = Pickable.__reduce_ex__


class WaitLock(SyncLock):
    is_WaitLock = True


class WithLock(SyncLock):
    is_WithLock = True


class FetchPrefetch(SyncData):
    is_Fetch = True


class FetchUpdate(SyncData):
    is_Fetch = True


class PrefetchUpdate(SyncData):
    is_Fetch = True


class WaitPrefetch(SyncData):
    pass


class Delete(SyncData):
    pass


def normalize_syncs(*args):
    if not args:
        return
    if len(args) == 1:
        return args[0]

    syncs = defaultdict(list)
    for _dict in args:
        for k, v in _dict.items():
            syncs[k].extend(v)

    syncs = {k: filter_ordered(v) for k, v in syncs.items()}

    for v in syncs.values():
        waitlocks = [i for i in v if i.is_WaitLock]
        withlocks = [i for i in v if i.is_WithLock]

        if waitlocks and withlocks:
            # We do not allow mixing up WaitLock and WithLock ops
            raise ValueError("Incompatible SyncOps")

    return syncs

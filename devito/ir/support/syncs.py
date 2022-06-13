"""
Synchronization operations inside the IR.
"""

from collections import defaultdict

from devito.tools import Pickable, filter_ordered

__all__ = ['WaitLock', 'ReleaseLock', 'WithLock', 'FetchUpdate', 'PrefetchUpdate',
           'normalize_syncs']


class SyncOp(Pickable):

    def __eq__(self, other):
        return (type(self) == type(other) and
                all(i == j for i, j in zip(self.args, other.args)))

    def __hash__(self):
        return hash((type(self).__name__,) + self.args)

    @property
    def args(self):
        return ()


class SyncLock(SyncOp):

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

    def __init__(self, dim, size, function, target, tstore, handle=None):
        self.dim = dim
        self.size = size
        self.function = function
        self.target = target
        self.tstore = tstore
        self.handle = handle

    def __repr__(self):
        return "%s<%s->%s:%s:%d>" % (self.__class__.__name__, self.function,
                                     self.target, self.dim, self.size)

    __str__ = __repr__

    @property
    def args(self):
        return (self.dim, self.size, self.function, self.target, self.tstore, self.handle)

    @property
    def dimensions(self):
        return self.function.dimensions

    # Pickling support
    _pickle_args = ['dim', 'size', 'function', 'target', 'tstore']
    _pickle_kwargs = ['handle']
    __reduce_ex__ = Pickable.__reduce_ex__


class WaitLock(SyncLock):
    pass


class WithLock(SyncLock):
    pass


class ReleaseLock(SyncLock):
    pass


class FetchUpdate(SyncData):
    pass


class PrefetchUpdate(SyncData):
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
        waitlocks = [s for s in v if isinstance(s, WaitLock)]
        withlocks = [s for s in v if isinstance(s, WithLock)]

        if waitlocks and withlocks:
            # We do not allow mixing up WaitLock and WithLock ops
            raise ValueError("Incompatible SyncOps")

    return syncs

"""
Synchronization operations inside the IR.
"""

from collections import defaultdict

from devito.data import FULL
from devito.tools import Pickable, filter_ordered

__all__ = ['WaitLock', 'ReleaseLock', 'WithLock', 'FetchUpdate', 'PrefetchUpdate',
           'normalize_syncs']


class SyncOp(Pickable):

    __rargs__ = ('function', 'handle')

    def __init__(self, function, handle):
        self.function = function
        self.handle = handle

    def __eq__(self, other):
        return (type(self) == type(other) and
                all(i == j for i, j in zip(self.args, other.args)))

    def __hash__(self):
        return hash((type(self).__name__,) + self.args)

    def __repr__(self):
        return "%s<%s>" % (self.__class__.__name__, self.handle)

    __str__ = __repr__

    @property
    def args(self):
        return (self.handle,)

    @property
    def lock(self):
        return self.handle.function

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class SyncCopyOut(SyncOp):

    @property
    def imask(self):
        ret = [self.handle.indices[d] if d.root in self.lock.locked_dimensions else FULL
               for d in self.function.dimensions]
        return tuple(ret)


class SyncCopyIn(SyncOp):

    __rargs__ = SyncOp.__rargs__ + ('dim', 'size', 'target', 'tstore')

    def __init__(self, function, handle, dim, size, target, tstore):
        super().__init__(function, handle)

        self.dim = dim
        self.size = size
        self.target = target
        self.tstore = tstore

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

    @property
    def imask(self):
        ret = [(self.tstore, self.size) if d.root is self.dim.root else FULL
               for d in self.dimensions]
        return tuple(ret)

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class WaitLock(SyncCopyOut):
    pass


class WithLock(SyncCopyOut):
    pass


class ReleaseLock(SyncCopyOut):
    pass


class FetchUpdate(SyncCopyIn):
    pass


class PrefetchUpdate(SyncCopyIn):
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

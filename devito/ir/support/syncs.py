"""
Synchronization operations inside the IR.
"""

from collections import defaultdict

from devito.data import FULL
from devito.tools import Pickable, filter_ordered
from devito.types import DimensionTuple

__all__ = ['WaitLock', 'ReleaseLock', 'WithLock', 'FetchUpdate', 'PrefetchUpdate',
           'normalize_syncs']


class IMask(DimensionTuple):
    pass


class SyncOp(Pickable):

    __rargs__ = ('handle', 'target')
    __rkwargs__ = ('tindex', 'function', 'findex', 'dim', 'size', 'origin')

    def __init__(self, handle, target, tindex=None, function=None, findex=None,
                 dim=None, size=1, origin=None):
        self.handle = handle
        self.target = target

        self.tindex = tindex
        self.function = function
        self.findex = findex
        self.dim = dim
        self.size = size
        self.origin = origin

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.handle == other.handle and
                self.target is other.target and
                self.tindex == other.tindex and
                self.function is other.function and
                self.findex == other.findex and
                self.dim is other.dim and
                self.size == other.size and
                self.origin == other.origin)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "%s<%s>" % (self.__class__.__name__, self.handle)

    __str__ = __repr__

    @property
    def lock(self):
        return self.handle.function

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class SyncCopyOut(SyncOp):

    def __repr__(self):
        return "%s<%s->%s>" % (self.__class__.__name__, self.target, self.function)

    __str__ = __repr__

    @property
    def imask(self):
        ret = [self.handle.indices[d] if d.root in self.lock.locked_dimensions else FULL
               for d in self.target.dimensions]
        return IMask(*ret, getters=self.target.dimensions, function=self.function,
                     findex=self.findex)


class SyncCopyIn(SyncOp):

    def __repr__(self):
        return "%s<%s->%s>" % (self.__class__.__name__, self.function, self.target)

    __str__ = __repr__

    @property
    def imask(self):
        ret = [(self.tindex, self.size) if d.root is self.dim.root else FULL
               for d in self.target.dimensions]
        return IMask(*ret, getters=self.target.dimensions, function=self.function,
                     findex=self.findex)


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

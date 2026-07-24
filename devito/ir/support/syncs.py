"""
Synchronization operations inside the IR.
"""

from collections import defaultdict
from functools import cached_property

from devito.data import FULL
from devito.tools import Pickable, as_tuple, filter_ordered, frozendict

from .utils import IMask

__all__ = [
    'InitArray',
    'Ops',
    'PrefetchUpdate',
    'ReleaseLock',
    'SnapIn',
    'SnapOut',
    'SyncArray',
    'WaitLock',
    'WithLock',
    'normalize_syncs',
]


class SyncOp(Pickable):

    """
    Metadata for a synchronization operation attached to a `Cluster` or `SyncSpot`.

    Parameters
    ----------
    handle : object
        The symbolic object identifying or controlling the operation, such as
        an entry in a `Lock`. May be None when no handle is required.
    target : AbstractFunction
        The `Function` whose access is synchronized. For buffered data movements,
        this is the compiler-generated buffer.
    tindex : Expr, optional
        The index into `target` involved in the operation.
    function : AbstractFunction, optional
        The original `Function` represented by a compiler-generated `target`. It
        is the source of a `SyncCopyIn` and the destination of a `SyncCopyOut`.
    findex : Expr, optional
        The index into `function` corresponding to `tindex`.
    dim : Dimension, optional
        The `Dimension` along which `tindex` and `findex` are defined.
    size : int, optional
        The extent associated with the operation along `dim`. Defaults to 1.
    origin : Indexed, optional
        The original `Indexed` access from which the operation was derived.
    gid : Stamp, optional
        The `Stamp` identifying the asynchronous task group to which the operation
        belongs.
    """

    __rargs__ = ('handle', 'target')
    __rkwargs__ = (
        'tindex', 'function', 'findex', 'dim', 'size', 'origin', 'gid'
    )

    def __init__(self, handle, target, tindex=None, function=None, findex=None,
                 dim=None, size=1, origin=None, gid=None):
        self.handle = handle
        self.target = target

        self.tindex = tindex
        self.function = function
        self.findex = findex
        self.dim = dim
        self.size = size
        self.origin = origin
        self.gid = gid

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.handle == other.handle and
                self.target is other.target and
                self.tindex == other.tindex and
                self.function is other.function and
                self.findex == other.findex and
                self.dim is other.dim and
                self.size == other.size and
                self.origin == other.origin and
                self.gid == other.gid)

    def __hash__(self):
        return hash((self.__class__, self.handle, self.target, self.tindex,
                     self.function, self.findex, self.dim, self.size, self.origin,
                     self.gid))

    def __repr__(self):
        return f"{self.__class__.__name__}<{self.handle.name}>"

    __str__ = __repr__

    @property
    def lock(self):
        try:
            return self.handle.function
        except AttributeError:
            return None

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class SyncCopyOut(SyncOp):

    def __repr__(self):
        return f"{self.__class__.__name__}<{self.target}->{self.function}>"

    __str__ = __repr__

    @property
    def imask(self):
        ret = []
        for d in self.target.dimensions:
            if d.root in self.lock.locked_dimensions:
                ret.append(self.handle.indices[d])
            else:
                ret.append(FULL)

        return IMask(*ret,
                     getters=self.target.dimensions,
                     function=self.function,
                     findex=self.findex,
                     mode='out')


class SyncCopyIn(SyncOp):

    def __repr__(self):
        return f"{self.__class__.__name__}<{self.function}->{self.target}>"

    __str__ = __repr__

    @property
    def imask(self):
        ret = []
        for d in self.target.dimensions:
            if d.root is self.dim.root:
                if self.target.is_regular:
                    ret.append((self.tindex, 1))
                else:
                    ret.append((0, 1))
            else:
                ret.append(FULL)

        return IMask(*ret,
                     getters=self.target.dimensions,
                     function=self.function,
                     findex=self.findex,
                     mode='in')


class WaitLock(SyncCopyOut):
    pass


class WithLock(SyncCopyOut):
    pass


class ReleaseLock(SyncCopyOut):
    pass


class InitArray(SyncCopyIn):
    pass


class SyncArray(SyncCopyIn):
    pass


class PrefetchUpdate(SyncCopyIn):
    pass


class SnapOut(SyncOp):
    pass


class SnapIn(SyncOp):
    pass


class Ops(frozendict):

    """
    A mapper {Dimension -> {SyncOps}}.
    """

    @property
    def dimensions(self):
        return tuple(self)

    def add(self, dims, ops):
        m = dict(self)
        for d in as_tuple(dims):
            m[d] = set(self.get(d, [])) | set(as_tuple(ops))
        return Ops(m)

    def update(self, ops):
        m = dict(self)
        for d, v in ops.items():
            m[d] = set(self.get(d, [])) | set(v)
        return Ops(m)

    def subs(self, mapper):
        return Ops({mapper.get(d, d): v for d, v in self.items()})

    def _get_sync(self, cls, dims=None):
        if dims is None:
            dims = list(self)
        for d in dims:
            for s in self.get(d, []):
                if isinstance(s, cls):
                    # NOTE: Remember there can only be one SyncOp of a given
                    # type per `Ops` object
                    return s
        return None

    @cached_property
    def initarray(self):
        return self._get_sync(InitArray)


def normalize_syncs(*args, strict=True):
    if not args:
        return {}

    syncs = defaultdict(list)
    for _dict in args:
        for k, v in _dict.items():
            syncs[k].extend(v)

    syncs = {k: tuple(filter_ordered(v)) for k, v in syncs.items()}

    if strict:
        for v in syncs.values():
            waitlocks = [s for s in v if isinstance(s, WaitLock)]
            withlocks = [s for s in v if isinstance(s, WithLock)]

            if waitlocks and withlocks:
                # We do not allow mixing up WaitLock and WithLock ops
                raise ValueError("Incompatible SyncOps")

    return Ops(syncs)

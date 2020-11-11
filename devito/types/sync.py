"""
Support types to implement asynchronous execution via e.g., threading,
device offloading, etc.
"""

import os
from collections import defaultdict
from ctypes import POINTER, c_void_p

import numpy as np
import sympy

from devito.parameters import configuration
from devito.tools import (Pickable, as_tuple, ctypes_to_cstr, dtype_to_ctype,
                          dtype_to_cstr, filter_ordered)
from devito.types.array import Array
from devito.types.basic import LocalObject
from devito.types.constant import Constant
from devito.types.dimension import CustomDimension

__all__ = ['NThreads', 'NThreadsNested', 'NThreadsNonaffine', 'NThreadsMixin',
           'ThreadID', 'Lock', 'WaitLock', 'WithLock', 'FetchWait', 'FetchWaitPrefetch',
           'Delete', 'STDThread', 'normalize_syncs']


class NThreadsMixin(object):

    is_PerfKnob = True

    def __new__(cls, **kwargs):
        name = kwargs.get('name', cls.name)
        value = cls.default_value()
        obj = Constant.__new__(cls, name=name, dtype=np.int32, value=value)
        obj.aliases = as_tuple(kwargs.get('aliases')) + (name,)
        return obj

    @property
    def _arg_names(self):
        return self.aliases

    def _arg_values(self, **kwargs):
        for i in self.aliases:
            if i in kwargs:
                return {self.name: kwargs.pop(i)}
        # Fallback: as usual, pick the default value
        return self._arg_defaults()


class NThreads(NThreadsMixin, Constant):

    name = 'nthreads'

    @classmethod
    def default_value(cls):
        return int(os.environ.get('OMP_NUM_THREADS',
                                  configuration['platform'].cores_physical))


class NThreadsNested(NThreadsMixin, Constant):

    name = 'nthreads_nested'

    @classmethod
    def default_value(cls):
        return configuration['platform'].threads_per_core


class NThreadsNonaffine(NThreads):

    name = 'nthreads_nonaffine'


class ThreadID(CustomDimension):

    def __new__(cls, nthreads):
        return CustomDimension.__new__(cls, name='tid', symbolic_size=nthreads)


class STDThread(LocalObject):

    dtype = type('std::thread', (c_void_p,), {})

    def __init__(self, name):
        self.name = name

    # Pickling support
    _pickle_args = ['name']


class Lock(Array):

    """
    An integer Array to synchronize accesses to a given object
    in a multithreaded context.
    """

    def __init_finalize__(self, *args, **kwargs):
        self._target = kwargs.pop('target')

        kwargs['scope'] = 'stack'
        kwargs['sharing'] = 'shared'
        super().__init_finalize__(*args, **kwargs)

    def __padding_setup__(self, **kwargs):
        # Bypass padding which is useless for locks
        kwargs['padding'] = 0
        return super().__padding_setup__(**kwargs)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return np.int32

    @property
    def target(self):
        return self._target

    @property
    def _C_typename(self):
        return 'volatile %s' % ctypes_to_cstr(POINTER(dtype_to_ctype(self.dtype)))

    @property
    def _C_typedata(self):
        return 'volatile %s' % dtype_to_cstr(self.dtype)

    @property
    def locked_dimensions(self):
        return set().union(*[d._defines for d in self.dimensions])


class SyncOp(sympy.Expr, Pickable):

    is_SyncLock = False
    is_SyncData = False

    is_WaitLock = False
    is_WithLock = False
    is_FetchWait = False
    is_FetchWaitPrefetch = False
    is_Delete = False

    def __new__(cls, handle):
        obj = sympy.Expr.__new__(cls, handle)
        obj.handle = handle
        return obj

    def __str__(self):
        return "%s<%s>" % (self.__class__.__name__, self.handle)

    __repr__ = __str__

    __hash__ = sympy.Basic.__hash__

    def __eq__(self, other):
        return type(self) == type(other) and self.args == other.args

    # Pickling support
    _pickle_args = ['handle']
    __reduce_ex__ = Pickable.__reduce_ex__


class SyncLock(SyncOp):

    is_SyncLock = True

    @property
    def lock(self):
        return self.handle.function

    @property
    def target(self):
        return self.lock.target


class SyncData(SyncOp):

    is_SyncData = True

    def __new__(cls, function, dim, fetch, size, direction=None):
        obj = sympy.Expr.__new__(cls, function, dim, fetch, size, direction)
        obj.function = function
        obj.dim = dim
        obj.fetch = fetch
        obj.size = size
        obj.direction = direction
        return obj

    def __str__(self):
        return "%s<%s:%s:%s:%d>" % (self.__class__.__name__, self.function,
                                    self.dim, self.fetch, self.size)

    __repr__ = __str__

    __hash__ = sympy.Basic.__hash__

    @property
    def dimensions(self):
        return self.function.dimensions

    # Pickling support
    _pickle_args = ['function', 'dim', 'fetch', 'size', 'direction']
    __reduce_ex__ = Pickable.__reduce_ex__


class WaitLock(SyncLock):
    is_WaitLock = True


class WithLock(SyncLock):
    is_WithLock = True


class FetchWait(SyncData):
    is_FetchWait = True


class FetchWaitPrefetch(SyncData):
    is_FetchWaitPrefetch = True


class Delete(SyncData):
    is_Delete = True


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

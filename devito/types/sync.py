"""
Support types to implement asynchronous execution via e.g., threading,
device offloading, etc.
"""

import os
from ctypes import POINTER, c_void_p

import numpy as np
import sympy

from devito.parameters import configuration
from devito.tools import Pickable, as_tuple, ctypes_to_cstr, dtype_to_ctype
from devito.types.array import Array
from devito.types.basic import LocalObject
from devito.types.constant import Constant
from devito.types.dimension import CustomDimension

__all__ = ['NThreads', 'NThreadsNested', 'NThreadsNonaffine', 'NThreadsMixin',
           'ThreadID', 'Lock', 'WaitLock', 'SetLock', 'UnsetLock', 'WaitThread',
           'WithThread', 'SyncData', 'DeleteData', 'STDThread']


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
    An integer Array usable as a lock in a multithreaded context
    """

    def __init_finalize__(self, *args, **kwargs):
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
    def _C_typename(self):
        return 'volatile %s' % ctypes_to_cstr(POINTER(dtype_to_ctype(self.dtype)))


class SyncOp(sympy.Expr, Pickable):

    is_WaitLock = False
    is_SetLock = False
    is_UnsetLock = False
    is_WaitThread = False
    is_WithThread = False
    is_SyncData = False
    is_DeleteData = False

    def __new__(cls, handle):
        obj = sympy.Expr.__new__(cls, handle)
        obj.handle = handle
        return obj

    def __str__(self):
        return "%s[%s]" % (self.__class__.__name__, self.handle)

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['handle']
    __reduce_ex__ = Pickable.__reduce_ex__


class WaitLock(SyncOp):
    is_WaitLock = True


class SetLock(SyncOp):
    is_SetLock = True


class UnsetLock(SyncOp):
    is_UnsetLock = True


class WaitThread(SyncOp):
    is_WaitThread = True


class WithThread(SyncOp):

    is_WithThread = True

    def __new__(cls, handle, sync_ops):
        obj = sympy.Expr.__new__(cls, handle, sync_ops)
        obj.handle = handle
        obj.sync_ops = as_tuple(sync_ops)
        return obj

    def __str__(self):
        return "%s[%s]<%s>" % (self.__class__.__name__, self.handle,
                               ','.join(str(i) for i in self.sync_ops))

    # Pickling support
    _pickle_args = ['handle', 'sync_ops']
    __reduce_ex__ = Pickable.__reduce_ex__


class SyncData(SyncOp):
    is_SyncData = True


class DeleteData(SyncOp):
    is_DeleteData = True

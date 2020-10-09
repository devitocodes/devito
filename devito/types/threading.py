"""
Support types to generate multithreaded code.
"""

import os
from ctypes import POINTER

import numpy as np

from devito.parameters import configuration
from devito.tools import Pickable, as_tuple, ctypes_to_cstr, dtype_to_ctype
from devito.types.array import Array
from devito.types.constant import Constant
from devito.types.dimension import CustomDimension

__all__ = ['NThreads', 'NThreadsNested', 'NThreadsNonaffine', 'NThreadsMixin',
           'ThreadID', 'Lock']


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

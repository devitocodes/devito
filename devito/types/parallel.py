"""
Support types for the generation of shared-memory parallel code. This module
contains basic types for threaded code (e.g., special symbols representing
the number of threads in a parallel region, objects such as locks to
implement thread synchronization, etc). Most of these objects are used internally
by the compiler.
"""

import os
from collections import defaultdict
from ctypes import c_void_p

from cached_property import cached_property
import numpy as np
import sympy

from devito.parameters import configuration
from devito.tools import Pickable, as_list, as_tuple, dtype_to_cstr, filter_ordered
from devito.types.array import Array, ArrayObject
from devito.types.basic import Symbol
from devito.types.constant import Constant
from devito.types.dimension import CustomDimension
from devito.types.misc import VolatileInt, c_volatile_int_p

__all__ = ['NThreads', 'NThreadsNested', 'NThreadsNonaffine', 'NThreadsMixin',
           'ThreadID', 'Lock', 'WaitLock', 'WithLock', 'FetchWait', 'FetchWaitPrefetch',
           'Delete', 'PThreadArray', 'SharedData', 'NPThreads', 'normalize_syncs']


class NThreadsMixin(object):

    is_PerfKnob = True

    def __new__(cls, **kwargs):
        name = kwargs.get('name', cls.name)
        value = cls.__value_setup__(**kwargs)
        obj = Constant.__new__(cls, name=name, dtype=np.int32, value=value)
        obj.aliases = as_tuple(kwargs.get('aliases')) + (name,)
        return obj

    @classmethod
    def __value_setup__(cls, **kwargs):
        try:
            return kwargs.pop('value')
        except KeyError:
            return cls.default_value()

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


class NPThreads(NThreadsMixin, Constant):

    name = 'npthreads'

    @classmethod
    def default_value(cls):
        return 1


class ThreadID(CustomDimension):

    def __new__(cls, nthreads):
        return CustomDimension.__new__(cls, name='tid', symbolic_size=nthreads)


class ThreadArray(ArrayObject):

    @classmethod
    def __indices_setup__(cls, **kwargs):
        try:
            return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
        except KeyError:
            nthreads = kwargs['npthreads']
            dim = CustomDimension(name='wi', symbolic_size=nthreads)
            return (dim,), (dim,)

    @property
    def dim(self):
        assert len(self.dimensions) == 1
        return self.dimensions[0]

    @property
    def index(self):
        if self.size == 1:
            return 0
        else:
            return self.dim

    @cached_property
    def symbolic_base(self):
        return Symbol(name=self.name, dtype=None)


class PThreadArray(ThreadArray):

    dtype = type('pthread_t', (c_void_p,), {})

    def __init_finalize__(self, *args, **kwargs):
        self.base_id = kwargs.pop('base_id')

        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return cls.dtype

    # Pickling support
    _pickle_kwargs = ThreadArray._pickle_kwargs + ['base_id']


class SharedData(ThreadArray):

    """
    An Array of structs, each struct containing data shared by one producer and
    one consumer thread.
    """

    _field_id = 'id'
    _field_flag = 'flag'

    _symbolic_id = Symbol(name=_field_id, dtype=np.int32)
    _symbolic_flag = VolatileInt(name=_field_flag)

    def __init_finalize__(self, *args, **kwargs):
        self.dynamic_fields = tuple(kwargs.pop('dynamic_fields', ()))

        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __pfields_setup__(cls, **kwargs):
        fields = as_list(kwargs.get('fields')) + [cls._symbolic_id, cls._symbolic_flag]
        return [(i._C_name, i._C_ctype) for i in fields]

    @cached_property
    def symbolic_id(self):
        return self._symbolic_id

    @cached_property
    def symbolic_flag(self):
        return self._symbolic_flag

    # Pickling support
    _pickle_kwargs = ThreadArray._pickle_kwargs + ['dynamic_fields']


class Lock(Array):

    """
    An integer Array to synchronize accesses to a given object
    in a multithreaded context.
    """

    def __init_finalize__(self, *args, **kwargs):
        self._target = kwargs.pop('target', None)

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
    def _C_ctype(self):
        return c_volatile_int_p

    @property
    def _C_typedata(self):
        return 'volatile %s' % dtype_to_cstr(self.dtype)

    @cached_property
    def locked_dimensions(self):
        return set().union(*[d._defines for d in self.dimensions])

    # Pickling support
    _pickle_kwargs = Array._pickle_kwargs + ['target']


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

"""
Support types for the generation of parallel code. This module contains types
for the generation of threaded code (e.g., special symbols representing
the number of threads in a parallel region, objects such as locks to
implement thread synchronization, etc) and device code (e.g., a special symbol
identifying a device attached to a node).
"""

import os
from ctypes import c_void_p

from cached_property import cached_property
import numpy as np

from devito.exceptions import InvalidArgument
from devito.parameters import configuration
from devito.tools import as_list, as_tuple, dtype_to_cstr
from devito.types.array import Array, ArrayObject
from devito.types.basic import Scalar, Symbol
from devito.types.dimension import CustomDimension
from devito.types.misc import Pointer, VolatileInt, c_volatile_int_p

__all__ = ['NThreads', 'NThreadsNested', 'NThreadsNonaffine', 'NThreadsBase',
           'DeviceID', 'ThreadID', 'Lock', 'PThreadArray', 'SharedData',
           'NPThreads', 'DeviceRM', 'DevicePointer']


class NThreadsBase(Scalar):

    is_Input = True
    is_PerfKnob = True

    def __new__(cls, *args, **kwargs):
        kwargs.setdefault('name', cls.name)
        kwargs['is_const'] = True
        return super().__new__(cls, **kwargs)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return np.int32

    @cached_property
    def default_value(self):
        return int(os.environ.get('OMP_NUM_THREADS',
                                  configuration['platform'].cores_physical))


class NThreads(NThreadsBase):

    name = 'nthreads'


class NThreadsNonaffine(NThreadsBase):

    name = 'nthreads_nonaffine'


class NThreadsNested(NThreadsBase):

    name = 'nthreads_nested'

    @property
    def default_value(self):
        return configuration['platform'].threads_per_core


class NPThreads(NThreadsBase):

    name = 'npthreads'

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, **kwargs)

        # Size of the thread pool
        obj.size = kwargs['size']

        return obj

    @property
    def default_value(self):
        return 1

    def _arg_values(self, **kwargs):
        if self.name in kwargs:
            v = kwargs.pop(self.name)
            if v < self.size:
                return {self.name: v}
            else:
                raise InvalidArgument("Illegal `%s=%d`. It must be `%s<%d`"
                                      % (self.name, v, self.name, self.size))
        else:
            return self._arg_defaults()

    # Pickling support
    _pickle_kwargs = NThreadsBase._pickle_kwargs + ['size']


class ThreadID(CustomDimension):

    def __new__(cls, nthreads):
        return CustomDimension.__new__(cls, name='tid', symbolic_size=nthreads)

    @property
    def nthreads(self):
        return self.symbolic_size

    _pickle_args = []
    _pickle_kwargs = ['nthreads']


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

    # Mandatory, or "static", fields
    _field_id = 'id'
    _field_deviceid = 'deviceid'
    _field_flag = 'flag'

    _symbolic_id = Symbol(name=_field_id, dtype=np.int32)
    _symbolic_deviceid = Symbol(name=_field_deviceid, dtype=np.int32)
    _symbolic_flag = VolatileInt(name=_field_flag)

    def __init_finalize__(self, *args, **kwargs):
        self.dynamic_fields = tuple(kwargs.pop('dynamic_fields', ()))

        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __pfields_setup__(cls, **kwargs):
        fields = as_list(kwargs.get('fields'))
        fields.extend([cls._symbolic_id, cls._symbolic_deviceid, cls._symbolic_flag])
        return [(i._C_name, i._C_ctype) for i in fields]

    @cached_property
    def symbolic_id(self):
        return self._symbolic_id

    @cached_property
    def symbolic_deviceid(self):
        return self._symbolic_deviceid

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


class DeviceSymbol(Scalar):

    is_Input = True
    is_PerfKnob = True

    def __new__(cls, *args, **kwargs):
        kwargs['name'] = cls.name
        kwargs['is_const'] = True
        return super().__new__(cls, **kwargs)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return np.int32


class DeviceID(DeviceSymbol):

    name = 'deviceid'

    @property
    def default_value(self):
        return -1


class DeviceRM(DeviceSymbol):

    name = 'devicerm'

    @property
    def default_value(self):
        return 1

    def _arg_values(self, **kwargs):
        try:
            # Enforce 1 or 0
            return {self.name: int(bool(kwargs[self.name]))}
        except KeyError:
            return self._arg_defaults()


class DevicePointer(Pointer):

    def __init_finalize__(self, *args, mapped=None, **kwargs):
        self._mapped = mapped

        super().__init_finalize__(*args, **kwargs)

    @property
    def mapped(self):
        return self._mapped

    @property
    def _C_typename(self):
        return self.mapped._C_typename

    # Pickling support
    _pickle_kwargs = Symbol._pickle_kwargs + ['mapped']

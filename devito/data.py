from __future__ import absolute_import

import abc
from functools import reduce
from operator import mul

import numpy as np
from sympy import Eq
import ctypes
from ctypes.util import find_library

from devito.logger import error
from devito.parameters import configuration
from devito.tools import as_tuple, numpy_to_ctypes
import devito

__all__ = ['Data', 'ALLOC_FLAT', 'ALLOC_NUMA_LOCAL', 'ALLOC_NUMA_ANY',
           'ALLOC_KNL_MCDRAM', 'ALLOC_KNL_DRAM', 'default_allocator']


class MemoryAllocator(object):

    """Abstract class defining the interface to memory allocators."""

    __metaclass__ = abc.ABCMeta

    @classmethod
    def initialized(cls):
        return cls.lib is not None

    def alloc(self, shape, dtype):
        """
        Allocate memory.

        :param shape: Shape of the array to allocate
        :param dtype: Numpy datatype to allocate. Default to np.float32

        :returns (pointer, c_pointer): The first element of the tuple is the reference
                                       that can be used to access the data as a ctypes
                                       object. The second element is the low-level
                                       reference that is needed only for the call to free.
        """
        size = int(reduce(mul, shape))
        ctype = numpy_to_ctypes(dtype)

        c_pointer = self._alloc_C_libcall(size, ctype)
        if c_pointer is None:
            error("Unable to allocate %d elements in memory", str(size))
            return (None, None)

        c_pointer = ctypes.cast(c_pointer, np.ctypeslib.ndpointer(dtype=dtype,
                                                                  shape=shape))
        pointer = np.ctypeslib.as_array(c_pointer, shape=shape)

        return (pointer, c_pointer)

    @abc.abstractmethod
    def _alloc_C_libcall(self):
        """
        Perform the actual memory allocation by calling a C function.

        .. note::

            This method should be implemented by a subclass.
        """
        return

    @abc.abstractmethod
    def free(self, c_pointer, size):
        """
        Free memory previously allocated with ``self.alloc``.

        :param c_pointer: The pointer to the memory region to be freed.
        :param size: The amount of memory to be freed, in bytes.

        .. note::

            This method should be implemented by a subclass.
        """
        return


class PosixAllocator(MemoryAllocator):

    """
    Memory allocator based on ``posix`` functions. The allocated memory is
    aligned to page boundaries.
    """

    handle = find_library('c')
    if handle is not None:
        lib = ctypes.CDLL(handle)
    else:
        lib = None

    def _alloc_C_libcall(self, size, ctype):
        c_bytesize = ctypes.c_ulong(size * ctypes.sizeof(ctype))
        c_pointer = ctypes.cast(ctypes.c_void_p(), ctypes.c_void_p)
        alignment = self.lib.getpagesize()
        ret = self.lib.posix_memalign(ctypes.byref(c_pointer), alignment, c_bytesize)
        if ret == 0:
            return c_pointer
        else:
            return None

    def free(self, c_pointer, size):
        self.lib.free(c_pointer)


class NumaAllocator(MemoryAllocator):

    """
    Memory allocator based on ``libnuma`` functions. The allocated memory is
    aligned to page boundaries. Through the argument ``node`` it is possible
    to specify a NUMA node in which memory allocation should be attempted first
    (will fall back to an arbitrary NUMA domain if not enough memory is available)

    :param node: Either an integer, indicating a specific NUMA node, or the special
                 keywords ``'local'`` or ``'any'``.
    """

    handle = find_library('numa')
    if handle is not None:
        lib = ctypes.CDLL(handle)
    else:
        lib = None
    if lib.numa_available() != -1:
        # We are indeed on a NUMA system
        # Allow the kernel to allocate memory on other NUMA nodes when there isn't
        # enough free on the target node
        lib.numa_set_bind_policy(0)
        # Required because numa_alloc* functions return pointers
        lib.numa_alloc_local.restype = ctypes.c_void_p
        lib.numa_alloc.restype = ctypes.c_void_p
    else:
        lib = None

    def __init__(self, node):
        super(NumaAllocator, self).__init__()
        self._node = node

    def _alloc_C_libcall(self, size, ctype):
        c_bytesize = ctypes.c_ulong(size * ctypes.sizeof(ctype))
        if isinstance(self._node, int):
            c_pointer = self.lib.numa_alloc_onnode(c_bytesize, self._node)
        elif self._node == 'local':
            c_pointer = self.lib.numa_alloc_local(c_bytesize)
        else:
            c_pointer = self.lib.numa_alloc(c_bytesize)
        if c_pointer == 0:
            return None
        else:
            return c_pointer

    def free(self, c_pointer, size):
        self.lib.numa_free(c_pointer, size)


ALLOC_FLAT = PosixAllocator()
ALLOC_KNL_DRAM = NumaAllocator(0)
ALLOC_KNL_MCDRAM = NumaAllocator(1)
ALLOC_NUMA_ANY = NumaAllocator('any')
ALLOC_NUMA_LOCAL = NumaAllocator('local')


def default_allocator():
    """
    Return a :class:`MemoryAllocator` for the architecture on which the
    process is running. Possible allocators are: ::

        * ALLOC_FLAT: Align memory to page boundaries using the posix function
                      ``posix_memalign``
        * ALLOC_NUMA_LOCAL: Allocate memory in the "closest" NUMA node. This only
                            makes sense on a NUMA architecture. Falls back to
                            allocation in an arbitrary NUMA node if there isn't
                            enough space.
        * ALLOC_NUMA_ANY: Allocate memory in an arbitrary NUMA node.
        * ALLOC_KNL_MCDRAM: On a Knights Landing platform, allocate memory in MCDRAM.
                            Falls back to DRAM if there isn't enough space.
        * ALLOC_KNL_DRAM: On a Knights Landing platform, allocate memory in DRAM.

    The default allocator is chosen based on the following algorithm: ::

        * If ``libnuma`` is not available on the system, return ALLOC_FLAT (though
          it typically is available, at least on relatively recent Linux distributions);
        * If on a Knights Landing platform (codename ``knl``, see ``print_defaults()``)
          return ALLOC_KNL_MCDRAM;
        * If on a multi-socket Intel Xeon platform, return ALLOC_NUMA_LOCAL;
        * Otherwise, return ALLOC_FLAT.
    """
    if NumaAllocator.initialized():
        if configuration['platform'] == 'knl':
            return ALLOC_KNL_MCDRAM
        else:
            return ALLOC_NUMA_LOCAL
    else:
        return ALLOC_FLAT


class Data(np.ndarray):

    """
    A special :class:`numpy.ndarray` allowing logical indexing.

    The type :class:`numpy.ndarray` is subclassed as indicated at: ::

        https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    :param shape: Shape of the domain region in grid points.
    :param dimensions: A tuple of :class:`Dimension`s, representing the dimensions
                       of the ``Data``.
    :param dtype: A ``numpy.dtype`` for the raw data.
    :param allocator: (Optional) a :class:`MemoryAllocator` to specialize memory
                      allocation. Defaults to ``ALLOC_FLAT``.

    .. note::

        This type supports logical indexing over modulo buffered dimensions.

    .. note::

        Any view or copy ``A`` created starting from ``self``, for instance via
        a slice operation or a universal function ("ufunc" in NumPy jargon), will
        still be of type :class:`Data`. However, if ``A``'s rank is different than
        ``self``'s rank, namely if ``A.ndim != self.ndim``, then the capability of
        performing logical indexing is lost.
    """

    def __new__(cls, shape, dimensions, dtype, allocator=ALLOC_FLAT):
        assert len(shape) == len(dimensions)
        ndarray, c_pointer = allocator.alloc(shape, dtype)
        obj = np.asarray(ndarray).view(cls)
        obj._allocator = allocator
        obj._c_pointer = c_pointer
        obj._modulo = tuple(True if i.is_Stepping else False for i in dimensions)
        return obj

    def __del__(self):
        if self._c_pointer is None:
            return
        self._allocator.free(self._c_pointer, self.nbytes)
        self._c_pointer = None

    def __array_finalize__(self, obj):
        # `self` is the newly created object
        # `obj` is the object from which `self` was created
        if obj is None:
            # `self` was created through __new__()
            return
        if type(obj) != Data or self.ndim != obj.ndim:
            self._modulo = tuple(False for i in range(self.ndim))
        else:
            self._modulo = obj._modulo
        # Views or references created via operations on `obj` do not get an
        # explicit reference to the C pointer (`_c_pointer`). This makes sure
        # that only one object (the "root" Data) will free the C-allocated memory
        self._c_pointer = None

    def __getitem__(self, index):
        index = self._convert_index(index)
        return super(Data, self).__getitem__(index)

    def __setitem__(self, index, val):
        index = self._convert_index(index)
        super(Data, self).__setitem__(index, val)

    def _convert_index(self, index):
        if isinstance(index, np.ndarray):
            # Advanced indexing, nothing special to do
            return index

        index = as_tuple(index)
        if len(index) > self.ndim:
            # Maybe user code is trying to add a new axis (see np.newaxis),
            # so the resulting array will have shape larger than `self`'s,
            # hence I can just let numpy deal with it, as by specification
            # we're gonna drop modulo indexing anyway
            return index

        # Index conversion
        wrapped = []
        for i, mod, use_modulo in zip(index, self.shape, self._modulo):
            if use_modulo is False:
                # Nothing special to do (no logical indexing)
                wrapped.append(i)
            elif isinstance(i, slice):
                if i.start is None:
                    start = i.start
                elif i.start >= 0:
                    start = i.start % mod
                else:
                    start = -(i.start % mod)
                if i.stop is None:
                    stop = i.stop
                elif i.stop >= 0:
                    stop = i.stop % (mod + 1)
                else:
                    stop = -(i.stop % (mod + 1))
                wrapped.append(slice(start, stop, i.step))
            elif isinstance(i, (tuple, list)):
                wrapped.append([k % mod for k in i])
            else:
                wrapped.append(i % mod)
        return wrapped[0] if len(index) == 1 else tuple(wrapped)

    def reset(self):
        """
        Set all grid entries to 0.
        """
        self[:] = 0.0


def first_touch(array):
    """
    Uses an Operator to initialize the given array in the same pattern that
    would later be used to access it.
    """
    devito.Operator(Eq(array, 0.))()

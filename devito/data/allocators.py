import abc
from functools import reduce
from operator import mul
import mmap
import os
import sys

import numpy as np
import ctypes
from ctypes.util import find_library

from devito.logger import logger
from devito.parameters import configuration
from devito.tools import dtype_to_ctype

__all__ = ['ALLOC_FLAT', 'ALLOC_NUMA_LOCAL', 'ALLOC_NUMA_ANY',
           'ALLOC_KNL_MCDRAM', 'ALLOC_KNL_DRAM', 'ALLOC_GUARD',
           'default_allocator']


class MemoryAllocator(object):

    """Abstract class defining the interface to memory allocators."""

    __metaclass__ = abc.ABCMeta

    is_Posix = False
    is_Numa = False

    _attempted_init = False
    lib = None

    guaranteed_alignment = 64
    """Guaranteed data alignment."""

    @classmethod
    def available(cls):
        if cls._attempted_init is False:
            cls.initialize()
            cls._attempted_init = True
        return cls.lib is not None

    @classmethod
    def initialize(cls):
        """
        Initialize the MemoryAllocator.

        Notes
        -----
        This method must be implemented by all subclasses of MemoryAllocator.
        """
        return

    def alloc(self, shape, dtype):
        """
        Allocate memory.

        Parameters
        ----------
        shape : tuple of ints
            Shape of the allocated array.
        dtype : numpy.dtype
            The data type of the raw data.

        Returns
        -------
        pointer, memfree_args
            The first element of the tuple is the reference that can be used to
            access the data as a ctypes object. The second element is an opaque
            object that is needed only for the "memfree" call.
        """
        size = int(reduce(mul, shape))
        ctype = dtype_to_ctype(dtype)

        c_pointer, memfree_args = self._alloc_C_libcall(size, ctype)
        if c_pointer is None:
            raise RuntimeError("Unable to allocate %d elements in memory", str(size))

        # cast to 1D array of the specified size
        ctype_1d = ctype * size
        buf = ctypes.cast(c_pointer, ctypes.POINTER(ctype_1d)).contents
        pointer = np.frombuffer(buf, dtype=dtype)
        # pointer.reshape should not be used here because it may introduce a copy
        # From https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html:
        # It is not always possible to change the shape of an array without copying the
        # data. If you want an error to be raised when the data is copied, you should
        # assign the new shape to the shape attribute of the array:
        pointer.shape = shape

        return (pointer, memfree_args)

    @abc.abstractmethod
    def _alloc_C_libcall(self, size, ctype):
        """
        Perform the actual memory allocation by calling a C function.  Should
        return a 2-tuple (c_pointer, memfree_args), where the free args are
        what is handed back to free() later to deallocate.

        Notes
        -----
        This method must be implemented by all subclasses of MemoryAllocator.
        """
        return

    @abc.abstractmethod
    def free(self, *args):
        """
        Free memory previously allocated with ``self.alloc``.

        Arguments are provided exactly as returned in the second element of the
        tuple returned by _alloc_C_libcall

        Notes
        -----
        This method must be implemented by all subclasses of MemoryAllocator.
        """
        return


class PosixAllocator(MemoryAllocator):

    """
    Memory allocator based on ``posix`` functions. The allocated memory is
    aligned to page boundaries.
    """

    is_Posix = True

    @classmethod
    def initialize(cls):
        handle = find_library('c')

        # Special case: on MacOS Big Sur any code that attempts to check
        # for dynamic library presence by looking for a file at a path
        # will fail. For this case, a static path is provided.
        if handle is None and os.name == "posix" and sys.platform == "darwin":
            handle = '/usr/lib/libc.dylib'

        if handle is not None:
            try:
                cls.lib = ctypes.CDLL(handle)
            except OSError:
                cls.lib = None

    def _alloc_C_libcall(self, size, ctype):
        if not self.available():
            raise RuntimeError("Couldn't find `libc`'s `posix_memalign` to "
                               "allocate memory")
        c_bytesize = ctypes.c_ulong(size * ctypes.sizeof(ctype))
        c_pointer = ctypes.cast(ctypes.c_void_p(), ctypes.c_void_p)
        alignment = self.lib.getpagesize()
        ret = self.lib.posix_memalign(ctypes.byref(c_pointer), alignment, c_bytesize)
        if ret == 0:
            return c_pointer, (c_pointer, )
        else:
            return None, None

    def free(self, c_pointer):
        self.lib.free(c_pointer)


class GuardAllocator(PosixAllocator):

    """
    Memory allocator based on ``posix`` functions. The allocated memory is
    aligned to page boundaries.  Additionally, it allocates extra memory
    before and after the data, and configures it so that an SEGV is thrown
    immediately if an out-of-bounds access occurs.

    Further, the remainder region of the last page (which cannot be
    protected), is poisoned with NaNs.
    """

    def __init__(self, padding_bytes=1024*1024):
        self.padding_bytes = padding_bytes

    def _alloc_C_libcall(self, size, ctype):
        if not self.available():
            raise RuntimeError("Couldn't find `libc`'s `posix_memalign` to "
                               "allocate memory")

        pagesize = self.lib.getpagesize()
        assert self.padding_bytes % pagesize == 0

        npages_pad = self.padding_bytes // pagesize
        nbytes_user = size * ctypes.sizeof(ctype)
        npages_user = (nbytes_user + pagesize - 1) // pagesize

        npages_alloc = 2*npages_pad + npages_user

        c_bytesize = ctypes.c_ulong(npages_alloc * pagesize)
        c_pointer = ctypes.cast(ctypes.c_void_p(), ctypes.c_void_p)
        alignment = self.lib.getpagesize()
        ret = self.lib.posix_memalign(ctypes.byref(c_pointer), alignment, c_bytesize)
        if ret != 0:
            return None, None

        # generate pointers to the left padding, the user data, and the right pad
        padleft_pointer = c_pointer
        c_pointer = ctypes.c_void_p(c_pointer.value + self.padding_bytes)
        padright_pointer = ctypes.c_void_p(c_pointer.value + npages_user * pagesize)

        # and set the permissions on the pad memory to 0 (no access)
        # if these fail, don't worry about failing the entire allocation
        c_padsize = ctypes.c_ulong(self.padding_bytes)
        if self.lib.mprotect(padleft_pointer, c_padsize, ctypes.c_int(0)):
            logger.warning("couldn't protect memory")
        if self.lib.mprotect(padright_pointer, c_padsize, ctypes.c_int(0)):
            logger.warning("couldn't protect memory")

        # if there is a multiple of 4 bytes left, use the code below to poison
        # the memory
        if nbytes_user % 4 == 0:
            poison_size = npages_user*pagesize - nbytes_user
            intp_type = ctypes.POINTER(ctypes.c_int)
            poison_ptr = ctypes.cast(ctypes.c_void_p(c_pointer.value + nbytes_user),
                                     intp_type)

            # for both float32 and float64, a sequence of -100 int32s represents NaNs,
            # at least on little-endian architectures.  It shouldn't matter what we
            # put in there, anyway
            for i in range(poison_size // 4):
                poison_ptr[i] = -100

        return c_pointer, (padleft_pointer, c_bytesize)

    def free(self, c_pointer, total_size):
        # unprotect it, since free() accesses it, I think...
        self.lib.mprotect(c_pointer, total_size,
                          ctypes.c_int(mmap.PROT_READ | mmap.PROT_WRITE))
        self.lib.free(c_pointer)


class NumaAllocator(MemoryAllocator):

    """
    Memory allocator based on ``libnuma`` functions. The allocated memory is
    aligned to page boundaries. Through the argument ``node`` it is possible
    to specify a NUMA node in which memory allocation should be attempted first
    (will fall back to an arbitrary NUMA domain if not enough memory is available)

    Parameters
    ----------
    node : int or str
        If an integer, it indicates a specific NUMA node. Otherwise, the two
        keywords ``local`` ("allocate on the local NUMA node") and ``any``
        ("allocate on any NUMA node with sufficient free memory") are accepted.
    """

    is_Numa = True

    @classmethod
    def initialize(cls):
        handle = find_library('numa')
        if handle is None:
            return
        lib = ctypes.CDLL(handle)
        if lib.numa_available() == -1:
            return
        # We are indeed on a NUMA system
        # Allow the kernel to allocate memory on other NUMA nodes when there isn't
        # enough free on the target node
        lib.numa_set_bind_policy(0)
        # Required because numa_alloc* functions return pointers
        lib.numa_alloc_onnode.restype = ctypes.c_void_p
        lib.numa_alloc_local.restype = ctypes.c_void_p
        lib.numa_alloc.restype = ctypes.c_void_p
        cls.lib = lib

    def __init__(self, node):
        super(NumaAllocator, self).__init__()
        self._node = node

    def _alloc_C_libcall(self, size, ctype):
        if not self.available():
            raise RuntimeError("Couldn't find `libnuma`'s `numa_alloc_*` to "
                               "allocate memory")

        if size == 0:
            # work around the fact that the allocator may return NULL when
            # the size is 0, and numpy does not like that
            c_bytesize = ctypes.c_ulong(1)
        else:
            c_bytesize = ctypes.c_ulong(size * ctypes.sizeof(ctype))

        if self.put_onnode:
            c_pointer = self.lib.numa_alloc_onnode(c_bytesize, self._node)
        elif self.put_local:
            c_pointer = self.lib.numa_alloc_local(c_bytesize)
        else:
            c_pointer = self.lib.numa_alloc(c_bytesize)

        # Note: even though restype was set above, ctypes returns a Python integer.
        # See https://stackoverflow.com/questions/17840144/
        # Edit: it apparently can return None, also!
        if c_pointer == 0 or c_pointer is None:
            return None, None
        else:
            # Convert it back to a void * - this is
            # _very_ important when later passing it to `numa_free`
            c_pointer = ctypes.c_void_p(c_pointer)
            return c_pointer, (c_pointer, c_bytesize)

    def free(self, c_pointer, c_bytesize):
        self.lib.numa_free(c_pointer, c_bytesize)

    @property
    def node(self):
        return self._node

    @property
    def put_onnode(self):
        return isinstance(self._node, int)

    @property
    def put_local(self):
        return self._node == 'local'


class ExternalAllocator(MemoryAllocator):

    """
    An ExternalAllocator is used to assign pre-existing user data to Functions.
    Thus, Devito does not allocate any memory.

    Parameters
    ----------
    array : array-like
        Any object exposing the buffer interface, such as a numpy.ndarray.

    Notes
    -------
    * Use ExternalAllocator and pass a reference to the external memory when
      creating a Function. This Function will now use this memory as its f.data.

    * If the data present in this external memory is valuable, provide a noop
      initialiser, or else Devito will reset it to 0.

    Example
    --------
    >>> from devito import Grid, Function
    >>> from devito.data.allocators import ExternalAllocator
    >>> import numpy as np
    >>> shape = (2, 2)
    >>> numpy_array = np.ones(shape, dtype=np.float32)
    >>> g = Grid(shape)
    >>> space_order = 0
    >>> f = Function(name='f', grid=g, space_order=space_order,
    ...      allocator=ExternalAllocator(numpy_array), initializer=lambda x: None)
    >>> f.data[0, 1] = 2
    >>> numpy_array
    array([[1., 2.],
           [1., 1.]], dtype=float32)
    """

    def __init__(self, numpy_array):
        self.numpy_array = numpy_array

    def alloc(self, shape, dtype):
        assert shape == self.numpy_array.shape, \
            "Provided array has shape %s. Expected %s" %\
            (str(self.numpy_array.shape), str(shape))
        assert dtype == self.numpy_array.dtype, \
            "Provided array has dtype %s. Expected %s" %\
            (str(self.numpy_array.dtype), str(dtype))

        return (self.numpy_array, None)


ALLOC_GUARD = GuardAllocator(1048576)
ALLOC_FLAT = PosixAllocator()
ALLOC_KNL_DRAM = NumaAllocator(0)
ALLOC_KNL_MCDRAM = NumaAllocator(1)
ALLOC_NUMA_ANY = NumaAllocator('any')
ALLOC_NUMA_LOCAL = NumaAllocator('local')

custom_allocators = {}
"""User-defined allocators."""


def register_allocator(name, allocator):
    """
    Register a custom MemoryAllocator.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a str, not `%s`" % type(name))
    if name in custom_allocators:
        raise ValueError("A MemoryAllocator for `%s` already exists" % name)
    if not isinstance(allocator, MemoryAllocator):
        raise TypeError("Expected a MemoryAllocator, not `%s`" % type(allocator))

    custom_allocators[name] = allocator


def infer_knl_mode():
    path = os.path.join('/sys', 'bus', 'node', 'devices', 'node1')
    return 'flat' if os.path.exists(path) else 'cache'


def default_allocator(name=None):
    """
    Return a MemoryAllocator for the underlying architecture.

        * ALLOC_GUARD: Only used in so-called "develop mode", to trigger SIGSEGV as
                       soon as OOB accesses are performed.
        * ALLOC_FLAT: Align memory to page boundaries using the posix function
                      `posix_memalign`.
        * ALLOC_NUMA_LOCAL: Allocate memory in the "closest" NUMA node. This only
                            makes sense on a NUMA architecture. Falls back to
                            allocation in an arbitrary NUMA node if there isn't
                            enough space.
        * ALLOC_NUMA_ANY: Allocate memory in an arbitrary NUMA node.
        * ALLOC_KNL_MCDRAM: On a Knights Landing platform, allocate memory in MCDRAM.
                            Falls back to DRAM if there isn't enough space.
        * ALLOC_KNL_DRAM: On a Knights Landing platform, allocate memory in DRAM.

    Custom allocators may be added with `register_allocator`.
    """
    if name is not None:
        try:
            return custom_allocators[name]
        except KeyError:
            pass

    if configuration['develop-mode']:
        return ALLOC_GUARD
    elif NumaAllocator.available():
        if configuration['platform'].name == 'knl' and infer_knl_mode() == 'flat':
            return ALLOC_KNL_MCDRAM
        else:
            return ALLOC_NUMA_LOCAL
    else:
        return ALLOC_FLAT

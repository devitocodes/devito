from __future__ import absolute_import

import abc
from functools import reduce
from operator import mul
import mmap

import numpy as np
import ctypes
from ctypes.util import find_library
from cached_property import cached_property

from devito.equation import Eq
from devito.parameters import configuration
from devito.tools import Tag, as_tuple, is_integer, numpy_to_ctypes, numpy_view_offsets
from devito.logger import logger
import devito

__all__ = ['ALLOC_FLAT', 'ALLOC_NUMA_LOCAL', 'ALLOC_NUMA_ANY',
           'ALLOC_KNL_MCDRAM', 'ALLOC_KNL_DRAM', 'ALLOC_GUARD']


class MemoryAllocator(object):

    """Abstract class defining the interface to memory allocators."""

    __metaclass__ = abc.ABCMeta

    is_Posix = False
    is_Numa = False

    _attempted_init = False
    lib = None

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

        .. note::

            This must be implemented by all subclasses of MemoryAllocator.
        """
        return

    def alloc(self, shape, dtype):
        """
        Allocate memory.

        :param shape: Shape of the array to allocate.
        :param dtype: Numpy datatype to allocate.

        :returns (pointer, free_handle): The first element of the tuple is the reference
                                         that can be used to access the data as a ctypes
                                         object. The second element is an opaque object
                                         that is needed only for the call to free.
        """
        size = int(reduce(mul, shape))
        ctype = numpy_to_ctypes(dtype)

        c_pointer, memfree_args = self._alloc_C_libcall(size, ctype)
        if c_pointer is None:
            raise RuntimeError("Unable to allocate %d elements in memory", str(size))

        c_pointer = ctypes.cast(c_pointer, np.ctypeslib.ndpointer(dtype=dtype,
                                                                  shape=shape))
        pointer = np.ctypeslib.as_array(c_pointer, shape=shape)

        return (pointer, memfree_args)

    @abc.abstractmethod
    def _alloc_C_libcall(self, size, ctype):
        """
        Perform the actual memory allocation by calling a C function.
        Should return a 2-tuple (c_pointer, memfree_args), where the free args
        are what is handed back to free() later to deallocate.

        .. note::

            This method should be implemented by a subclass.
        """
        return

    @abc.abstractmethod
    def free(self, *args):
        """
        Free memory previously allocated with ``self.alloc``.

        Arguments are provided exactly as returned in the second
        element of the tuple returned by _alloc_C_libcall

        .. note::

            This method should be implemented by a subclass.
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
        if handle is not None:
            cls.lib = ctypes.CDLL(handle)

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

    :param node: Either an integer, indicating a specific NUMA node, or the special
                 keywords ``'local'`` or ``'any'``.
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
        c_bytesize = ctypes.c_ulong(size * ctypes.sizeof(ctype))
        if self.put_onnode:
            c_pointer = self.lib.numa_alloc_onnode(c_bytesize, self._node)
        elif self.put_local:
            c_pointer = self.lib.numa_alloc_local(c_bytesize)
        else:
            c_pointer = self.lib.numa_alloc(c_bytesize)

        # note!  even though restype was set above, ctypes returns a
        # python integer.
        # See https://stackoverflow.com/questions/17840144/
        if c_pointer == 0:
            return None, None
        else:
            # Convert it back to a void * - this is
            # _very_ important when later # passing it to numa_free
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


ALLOC_GUARD = GuardAllocator(1048576)
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

        * If running in DEVELOP mode (env var DEVITO_DEVELOP), return ALLOC_FLAT;
        * If ``libnuma`` is not available on the system, return ALLOC_FLAT (though
          it typically is available, at least on relatively recent Linux distributions);
        * If on a Knights Landing platform (codename ``knl``, see ``print_defaults()``)
          return ALLOC_KNL_MCDRAM;
        * If on a multi-socket Intel Xeon platform, return ALLOC_NUMA_LOCAL;
        * In all other cases, return ALLOC_FLAT.
    """
    if configuration['develop-mode']:
        return ALLOC_GUARD
    elif NumaAllocator.available():
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

    :param shape: Shape of the array in grid points.
    :param dimensions: The array :class:`Dimension`s.
    :param dtype: A ``numpy.dtype`` for the raw data.
    :param decomposition: (Optional) a mapper from :class:`Dimension`s in
                          ``dimensions`` to :class:`Decomposition`s, which
                          describe how the Data is distributed over a set
                          of processes. The Decompositions will be used to
                          translate global array indices into local indices.
                          The local indices are relative to the calling process.
                          This is only relevant in the case of distributed
                          memory execution (via MPI).
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

    def __new__(cls, shape, dimensions, dtype, decomposition=None, allocator=ALLOC_FLAT):
        assert len(shape) == len(dimensions)
        ndarray, memfree_args = allocator.alloc(shape, dtype)
        obj = np.asarray(ndarray).view(cls)
        obj._allocator = allocator
        obj._memfree_args = memfree_args
        obj._decomposition = tuple((decomposition or {}).get(i) for i in dimensions)
        obj._modulo = tuple(True if i.is_Stepping else False for i in dimensions)

        # By default, the indices used to access array values are interpreted as
        # local indices
        obj._glb_indexing = False

        # This cannot be a property, as Data objects constructed from this
        # object might not have any `decomposition`, but they would still be
        # distributed. Hence, in `__array_finalize__` we must copy this value
        obj._is_distributed = any(i is not None for i in obj._decomposition)

        # Sanity check -- A Dimension can't be at the same time modulo-iterated
        # and MPI-distributed
        assert all(i is None for i, j in zip(obj._decomposition, obj._modulo)
                   if j is True)

        return obj

    def __del__(self):
        if self._memfree_args is None:
            return
        self._allocator.free(*self._memfree_args)
        self._memfree_args = None

    def __array_finalize__(self, obj):
        # `self` is the newly created object
        # `obj` is the object from which `self` was created
        if obj is None:
            # `self` was created through __new__()
            return
        if type(obj) != Data:
            self._modulo = tuple(False for i in range(self.ndim))
            self._decomposition = (None,)*self.ndim
            self._glb_indexing = False
            self._is_distributed = False
        elif self.ndim != obj.ndim:
            self._modulo = tuple(False for i in range(self.ndim))
            self._decomposition = (None,)*self.ndim
            self._glb_indexing = False
            self._is_distributed = obj._is_distributed
        else:
            self._modulo = obj._modulo
            self._glb_indexing = obj._glb_indexing
            self._is_distributed = obj._is_distributed
            # Note: the decomposition needs to be updated based on the extent
            # of the view
            offsets = numpy_view_offsets(self, obj._datatop)
            assert len(obj._decomposition) == len(offsets)
            self._decomposition = tuple(i.reshape(-lofs, -rofs) for i, (lofs, rofs)
                                        in zip(obj._datatop._decomposition, offsets))
        # Views or references created via operations on `obj` do not get an
        # explicit reference to the underlying data (`_memfree_args`). This makes sure
        # that only one object (the "root" Data) will free the C-allocated memory
        self._memfree_args = None

    @property
    def _local(self):
        """Return a view of ``self`` with disabled global indexing."""
        ret = self.view()
        ret._glb_indexing = False
        return ret

    @property
    def _global(self):
        """Return a view of ``self`` with enabled global indexing."""
        ret = self.view()
        ret._glb_indexing = True
        return ret

    @cached_property
    def _datatop(self):
        if isinstance(self.base, Data):
            return self.base._datatop
        else:
            assert isinstance(self.base, np.ndarray)
            return self

    def __repr__(self):
        return super(Data, self._local).__repr__()

    def __getitem__(self, glb_index):
        loc_index = self._convert_index(glb_index)
        if loc_index is NONLOCAL:
            return None
        else:
            return super(Data, self).__getitem__(loc_index)

    def __setitem__(self, glb_index, val):
        loc_index = self._convert_index(glb_index)
        if loc_index is NONLOCAL:
            # no-op
            return
        elif np.isscalar(val):
            pass
        elif isinstance(val, Data) and val._is_distributed:
            if self._is_distributed:
                # `val` is distributed, `self` is distributed -> local set
                # TODO: shapes and distributions must match
                pass
            else:
                # `val` is distributed, `self` is replicated -> gatherall-like
                raise NotImplementedError
        elif isinstance(val, np.ndarray):
            if self._is_distributed:
                # `val` is replicated, `self` is distributed -> `val` gets distributed
                if self._glb_indexing:
                    glb_index = index_normalize(glb_index)
                    glb_index = glb_index + (slice(None),)*(self.ndim - len(glb_index))
                    val_index = [index_apply_offset(dec(), i) if dec is not None else i
                                 for dec, i in zip(self._decomposition, glb_index)]
                    val = val[val_index]
            else:
                # `val` is replicated`, `self` is replicated -> plain ndarray.__setitem__
                pass
        elif isinstance(val, (tuple, list)):
            if self._is_distributed and configuration['mpi']:
                raise NotImplementedError("Cannot set `Data` values with tuples or lists "
                                          "when the object is distributed over processes")
        else:
            raise ValueError("Cannot insert obj of type `%s` into a Data" % type(val))

        # Finally, perform the actual __setitem__
        super(Data, self).__setitem__(loc_index, val)

    def _convert_index(self, index):
        index = index_normalize(index)

        if len(index) > self.ndim:
            # Maybe user code is trying to add a new axis (see np.newaxis),
            # so the resulting array will be higher dimensional than `self`,
            if self._is_distributed:
                raise ValueError("Cannot increase the dimensionality of distributed Data")
            else:
                # As by specification, we are forced to ignore modulo indexing
                return index

        ret = []
        for i, s, mod, dec in zip(index, self.shape, self._modulo, self._decomposition):
            if mod is True:
                # Need to wrap index based on modulo
                v = index_apply_modulo(i, s)
            elif self._glb_indexing is True and dec is not None:
                # Need to convert the user-provided global indices into local
                # indices. This has no effect if MPI is not used.
                v = index_glb_to_loc(i, dec)
            else:
                v = i

            # Handle non-local, yet globally legal, indices
            v = index_handle_oob(v)

            ret.append(v)

        # Deal with NONLOCAL accesses
        if NONLOCAL in ret:
            if len(ret) == self.ndim and index_is_basic(ret):
                # Caller expecting a scalar -- it will eventually get None
                ret = [NONLOCAL]
            else:
                # Caller expecting an array -- it will eventually get a 0-length array
                ret = [slice(-1, -2) if i is NONLOCAL else i for i in ret]

        return ret[0] if len(ret) == 1 else tuple(ret)

    def reset(self):
        """
        Set all Data entries to 0.
        """
        self[:] = 0.0


class Index(Tag):
    pass
NONLOCAL = Index('nonlocal')  # noqa


def index_normalize(index):
    if isinstance(index, np.ndarray):
        # Advanced indexing mode
        index = (index,)
    return as_tuple(index)


def index_is_basic(index):
    return all(is_integer(i) or (i is NONLOCAL) for i in index)


def index_apply_offset(index, ofs):
    if isinstance(ofs, slice):
        ofs = ofs.start
    if ofs is None:
        ofs = 0
    if not is_integer(ofs):
        raise ValueError("Cannot apply offset of type `%s`" % type(ofs))

    if is_integer(index):
        return index - ofs
    elif isinstance(index, slice):
        return slice(index.start - ofs, index.stop - ofs, index.step)
    elif isinstance(index, (tuple, list)):
        return [i - ofs for i in index]
    elif isinstance(index, np.ndarray):
        return index - ofs
    else:
        raise ValueError("Cannot apply offset to index of type `%s`" % type(index))


def index_apply_modulo(index, modulo):
    if is_integer(index):
        return index % modulo
    elif isinstance(index, slice):
        if index.start is None:
            start = index.start
        elif index.start >= 0:
            start = index.start % modulo
        else:
            start = -(index.start % modulo)
        if index.stop is None:
            stop = index.stop
        elif index.stop >= 0:
            stop = index.stop % (modulo + 1)
        else:
            stop = -(index.stop % (modulo + 1))
        return slice(start, stop, index.step)
    elif isinstance(index, (tuple, list)):
        return [i % modulo for i in index]
    elif isinstance(index, np.ndarray):
        return index
    else:
        raise ValueError("Cannot apply modulo to index of type `%s`" % type(index))


def index_glb_to_loc(index, decomposition):
    if is_integer(index):
        return decomposition(index)
    elif isinstance(index, slice):
        return slice(*decomposition((index.start, index.stop)), index.step)
    elif isinstance(index, (tuple, list)):
        return [decomposition(i) for i in index]
    elif isinstance(index, np.ndarray):
        return np.vectorize(lambda i: decomposition(i))(index)
    else:
        raise ValueError("Cannot convert global index of type `%s` into a local index"
                         % type(index))


def index_handle_oob(index):
    # distributed.convert_index returns None when the index is globally
    # legal, but out-of-bounds for the calling MPI rank
    if index is None:
        return NONLOCAL
    elif isinstance(index, (tuple, list)):
        return [i for i in index if i is not None]
    elif isinstance(index, np.ndarray):
        return np.delete(index, np.where(index == None))  # noqa
    else:
        return index


def first_touch(array):
    """
    Uses an Operator to initialize the given array in the same pattern that
    would later be used to access it.
    """
    devito.Operator(Eq(array, 0.))()

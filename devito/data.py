from __future__ import absolute_import

import abc
from functools import reduce
from operator import mul
import mmap

import numpy as np
import ctypes
from ctypes.util import find_library

from devito.equation import Eq
from devito.parameters import configuration
from devito.tools import Tag, as_tuple, is_integer, numpy_to_ctypes
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

        # Saves the last index used in `__getitem__`. This allows `__array_finalize__`
        # to reconstruct information about the computed view (e.g., `decomposition`)
        obj._index_stash = None

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

        self._index_stash = None

        # Views or references created via operations on `obj` do not get an
        # explicit reference to the underlying data (`_memfree_args`). This makes sure
        # that only one object (the "root" Data) will free the C-allocated memory
        self._memfree_args = None

        if type(obj) != Data:
            # Definitely from view casting
            self._glb_indexing = False
            self._is_distributed = False
            self._modulo = tuple(False for i in range(self.ndim))
            self._decomposition = (None,)*self.ndim
        elif obj._index_stash is not None:
            # From `__getitem__`
            self._glb_indexing = obj._glb_indexing
            self._is_distributed = obj._is_distributed
            idx = obj._normalize_index(obj._index_stash)
            self._modulo = tuple(m for i, m in zip(idx, obj._modulo) if not is_integer(i))
            decomposition = []
            for i, dec in zip(idx, obj._decomposition):
                if is_integer(i):
                    continue
                elif dec is None:
                    decomposition.append(None)
                else:
                    decomposition.append(dec.reshape(i))
            self._decomposition = tuple(decomposition)
        else:
            self._glb_indexing = obj._glb_indexing
            self._is_distributed = obj._is_distributed
            if self.ndim == obj.ndim:
                # E.g., from a ufunc, such as `np.add`
                self._modulo = obj._modulo
                self._decomposition = obj._decomposition
            else:
                # E.g., from a reduction operation such as `np.mean` or `np.all`
                self._modulo = tuple(False for i in range(self.ndim))
                self._decomposition = (None,)*self.ndim

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

    def __repr__(self):
        return super(Data, self._local).__repr__()

    def __getitem__(self, glb_idx):
        loc_idx = self._convert_index(glb_idx)
        if loc_idx is NONLOCAL:
            # Caller expects a scalar. However, `glb_idx` doesn't belong to
            # self's data partition, so None is returned
            return None
        else:
            self._index_stash = glb_idx  # Will be popped in `__array_finalize__`
            retval = super(Data, self).__getitem__(loc_idx)
            self._index_stash = None
            return retval

    def __setitem__(self, glb_idx, val):
        loc_idx = self._convert_index(glb_idx)
        if loc_idx is NONLOCAL:
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
                    val_idx = self._normalize_index(glb_idx)
                    val_idx = [index_dist_to_repl(i, dec) for i, dec in
                               zip(val_idx, self._decomposition)]
                    if NONLOCAL in val_idx:
                        # no-op
                        return
                    val_idx = [i for i in val_idx if i is not PROJECTED]
                    val = val[val_idx]
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
        # Note: we pass `glb_idx`, rather than `loc_idx`, as __setitem__ calls
        # `__getitem__`, which in turn expects a global index
        super(Data, self).__setitem__(glb_idx, val)

    def _normalize_index(self, idx):
        if isinstance(idx, np.ndarray):
            # Advanced indexing mode
            idx = (idx,)
        idx = as_tuple(idx)
        idx = idx + (slice(None),)*(self.ndim - len(idx))
        return idx

    def _convert_index(self, glb_idx):
        glb_idx = self._normalize_index(glb_idx)

        if len(glb_idx) > self.ndim:
            # Maybe user code is trying to add a new axis (see np.newaxis),
            # so the resulting array will be higher dimensional than `self`,
            if self._is_distributed:
                raise ValueError("Cannot increase the dimensionality of distributed Data")
            else:
                # As by specification, we are forced to ignore modulo indexing
                return glb_idx

        loc_idx = []
        for i, s, mod, dec in zip(glb_idx, self.shape, self._modulo, self._decomposition):
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

            loc_idx.append(v)

        # Deal with NONLOCAL accesses
        if NONLOCAL in loc_idx:
            if len(loc_idx) == self.ndim and index_is_basic(loc_idx):
                # Caller expecting a scalar -- it will eventually get None
                loc_idx = [NONLOCAL]
            else:
                # Caller expecting an array -- it will eventually get a 0-length array
                loc_idx = [slice(-1, -2) if i is NONLOCAL else i for i in loc_idx]

        return loc_idx[0] if len(loc_idx) == 1 else tuple(loc_idx)

    def reset(self):
        """
        Set all Data entries to 0.
        """
        self[:] = 0.0


class Index(Tag):
    pass
NONLOCAL = Index('nonlocal')  # noqa
PROJECTED = Index('projected')


def index_is_basic(idx):
    return all(is_integer(i) or (i is NONLOCAL) for i in idx)


def index_apply_modulo(idx, modulo):
    if is_integer(idx):
        return idx % modulo
    elif isinstance(idx, slice):
        if idx.start is None:
            start = idx.start
        elif idx.start >= 0:
            start = idx.start % modulo
        else:
            start = -(idx.start % modulo)
        if idx.stop is None:
            stop = idx.stop
        elif idx.stop >= 0:
            stop = idx.stop % (modulo + 1)
        else:
            stop = -(idx.stop % (modulo + 1))
        return slice(start, stop, idx.step)
    elif isinstance(idx, (tuple, list)):
        return [i % modulo for i in idx]
    elif isinstance(idx, np.ndarray):
        return idx
    else:
        raise ValueError("Cannot apply modulo to index of type `%s`" % type(idx))


def index_dist_to_repl(idx, decomposition):
    """
    Convert a distributed array index a replicated array index.
    """
    if is_integer(idx):
        return PROJECTED

    if decomposition is None:
        return idx

    # Derive shift value
    value = idx.start if isinstance(idx, slice) else idx
    if value is None:
        value = 0
    elif not is_integer(value):
        raise ValueError("Cannot derive shift value from type `%s`" % type(value))

    # Convert into absolute local index
    idx = decomposition.convert_index(idx, rel=False)

    if idx is None:
        return NONLOCAL
    elif isinstance(idx, (tuple, list)):
        return [i - value for i in idx]
    elif isinstance(idx, np.ndarray):
        return idx - value
    elif isinstance(idx, slice):
        return slice(idx.start - value, idx.stop - value, idx.step)
    else:
        raise ValueError("Cannot apply shift to type `%s`" % type(idx))


def index_glb_to_loc(idx, decomposition):
    """
    Convert a global index into a local index.
    """
    if is_integer(idx) or isinstance(idx, slice):
        return decomposition(idx)
    elif isinstance(idx, (tuple, list)):
        return [decomposition(i) for i in idx]
    elif isinstance(idx, np.ndarray):
        return np.vectorize(lambda i: decomposition(i))(idx)
    else:
        raise ValueError("Cannot convert global index of type `%s` into a local index"
                         % type(idx))


def index_handle_oob(idx):
    # distributed.convert_index returns None when the index is globally
    # legal, but out-of-bounds for the calling MPI rank
    if idx is None:
        return NONLOCAL
    elif isinstance(idx, (tuple, list)):
        return [i for i in idx if i is not None]
    elif isinstance(idx, np.ndarray):
        return np.delete(idx, np.where(idx == None))  # noqa
    else:
        return idx


def first_touch(array):
    """
    Uses an Operator to initialize the given array in the same pattern that
    would later be used to access it.
    """
    devito.Operator(Eq(array, 0.))()

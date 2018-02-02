from __future__ import absolute_import

import ctypes
from ctypes.util import find_library
from functools import reduce
from operator import mul

import numpy as np
from sympy import Eq

from devito.logger import error
from devito.tools import as_tuple, numpy_to_ctypes
import devito


class Data(np.ndarray):

    """
    A special :class:`numpy.ndarray` allowing logical indexing.

    The type :class:`numpy.ndarray` is subclassed as indicated at: ::

        https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    :param shape: Shape of the domain region in grid points.
    :param dimensions: A tuple of :class:`Dimension`s, representing the dimensions
                       of the ``Data``.
    :param dtype: A ``numpy.dtype`` for the raw data.

    .. note::

        This type supports logical indexing over modulo buffered dimensions.

    .. note::

        Any view or copy ``A`` created starting from ``self``, for instance via
        a slice operation or a universal function ("ufunc" in NumPy jargon), will
        still be of type :class:`Data`. However, if ``A``'s rank is different than
        ``self``'s rank, namely if ``A.ndim != self.ndim``, then the capability of
        performing logical indexing is lost.
    """

    def __new__(cls, shape, dimensions, dtype):
        assert len(shape) == len(dimensions)
        ndarray, c_pointer = malloc_aligned(shape, dtype)
        obj = np.asarray(ndarray).view(cls)
        obj._c_pointer = c_pointer
        obj._modulo = tuple(True if i.is_Stepping else False for i in dimensions)
        return obj

    def __del__(self):
        if self._c_pointer is None:
            return
        free(self._c_pointer)
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


"""
Pre-load ``libc`` to explicitly manage C memory
"""
libc = ctypes.CDLL(find_library('c'))


def malloc_aligned(shape, dtype=np.float32, alignment=None):
    """
    Allocate memory using the C function ``malloc_aligned``.

    :param shape: Shape of the array to allocate
    :param dtype: Numpy datatype to allocate. Default to np.float32
    :param alignment: number of bytes to align to. Defaults to
                      page size if not set.

    :returns (pointer, c_pointer): the first element of the tuple is the reference
                                   that can be used to access the data as a ctypes
                                   object. The second element is the low-level
                                   reference that is needed only for the call to free.
    """
    c_pointer = ctypes.cast(ctypes.c_void_p(), ctypes.POINTER(ctypes.c_float))
    arraysize = int(reduce(mul, shape))
    ctype = numpy_to_ctypes(dtype)
    if alignment is None:
        alignment = libc.getpagesize()

    ret = libc.posix_memalign(ctypes.byref(c_pointer), alignment,
                              ctypes.c_ulong(arraysize * ctypes.sizeof(ctype)))
    if not ret == 0:
        error("Unable to allocate memory for shape %s", str(shape))
        return None

    c_pointer = ctypes.cast(c_pointer, np.ctypeslib.ndpointer(dtype=dtype, shape=shape))

    pointer = np.ctypeslib.as_array(c_pointer, shape=shape)
    return (pointer, c_pointer)


def free(c_pointer):
    """
    Use the C function free to free the memory allocated for the
    given pointer.
    """
    libc.free(c_pointer)


def first_touch(array):
    """
    Uses an Operator to initialize the given array in the same pattern that
    would later be used to access it.
    """
    devito.Operator(Eq(array, 0.))()

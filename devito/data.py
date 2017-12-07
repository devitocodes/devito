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
    The type :class:`numpy.ndarray` is subclassed as indicated at: ::

        https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    The new instance takes an existing ndarray as input, casts it to one of
    type ``Data``, and adds extra attributes (hence the need for subclassing)

    :param shape: Shape of the domain region in grid points.
    :param dimensions: A tuple of :class:`Dimension`s, representing the dimensions
                       of the ``Data``.
    :param halo: An integer indicating the extent of the halo region.
    :param dtype: A ``numpy.dtype`` for the raw data.

    .. note::

        This type supports logic indexing over modulo buffered dimensions.

    .. note::

        Any view or copy ``A`` created starting from ``self``, for instance via
        a slice operation or a universal function ("ufunc" in NumPy jargon), will
        still be of type :class:`Data`. However, if ``A`` is a view and its rank
        is lower than that of ``self``, namely ``A.ndim < self.ndim``, then the
        ``modulo`` attribute is dropped. A suitable exception is then raised if
        user code seems to attempt accessing data through modulo iteration on
        such a contracted view.
    """

    def __new__(cls, shape, dimensions, halo, dtype):
        ndarray, data_pointer = malloc_aligned(shape, dtype)
        obj = np.asarray(ndarray).view(cls)
        obj.data_pointer = data_pointer
        obj.halo = halo
        obj.modulo = tuple(i.modulo if i.is_Stepping else None for i in dimensions)
        return obj

    def __del__(self):
        if self.data_pointer is not None:
            return
        free(self.data_pointer)
        self.data_pointer = None

    def __array_finalize__(self, obj):
        if type(obj) != Data:
            return
        # `self` is the newly created object
        # `obj` is the object from which `self` was created
        if self.ndim == obj.ndim:
            self.modulo = getattr(obj, 'modulo', tuple(None for i in range(self.ndim)))
        else:
            self.modulo = tuple(None for i in range(self.ndim))
        self.halo = getattr(obj, 'halo', None)
        # Views or references created via operations on `obj` do not get
        # an explicit reference to the C pointer (`data_pointer`). This makes
        # sure that only one object (the "root" Data) will free the C-allocated
        self.data_pointer = None

    def __getitem__(self, index):
        index = self._convert_index(index)
        return super(Data, self).__getitem__(index)

    def __setitem__(self, index, val):
        index = self._convert_index(index)
        super(Data, self).__setitem__(index, val)

    def _convert_index(self, index):
        if isinstance(index, np.ndarray):
            # Using a mask array, nothing we really have to do
            return index
        else:
            index = as_tuple(index)
            wrapped = []
            for i, mod in zip(index, self.modulo):
                if mod is None:
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

    @property
    def with_halo(self):
        # TODO: Implement contextually to the domain-allocation switch
        raise NotImplementedError

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

    :returns (pointer, data_pointer): the first element of the tuple is the reference
                                      that can be used to access the data as a ctypes
                                      object. The second element is the low-level
                                      reference that is needed only for the call to free.
    """
    data_pointer = ctypes.cast(ctypes.c_void_p(), ctypes.POINTER(ctypes.c_float))
    arraysize = int(reduce(mul, shape))
    ctype = numpy_to_ctypes(dtype)
    if alignment is None:
        alignment = libc.getpagesize()

    ret = libc.posix_memalign(ctypes.byref(data_pointer), alignment,
                              ctypes.c_ulong(arraysize * ctypes.sizeof(ctype)))
    if not ret == 0:
        error("Unable to allocate memory for shape %s", str(shape))
        return None

    data_pointer = ctypes.cast(data_pointer,
                               np.ctypeslib.ndpointer(dtype=dtype, shape=shape))

    pointer = np.ctypeslib.as_array(data_pointer, shape=shape)
    return (pointer, data_pointer)


def free(pointer):
    """
    Use the C function free to free the memory allocated for the
    given pointer.
    """
    libc.free(pointer)


def first_touch(array):
    """
    Uses an Operator to initialize the given array in the same pattern that
    would later be used to access it.
    """
    exp_init = [Eq(array.indexed[array.indices], 0)]
    op = devito.Operator(exp_init)
    op.apply()

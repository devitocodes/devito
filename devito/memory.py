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

"""
Pre-load ``libc`` to explicitly manage C memory
"""
libc = ctypes.CDLL(find_library('c'))


class CMemory(object):

    def __init__(self, shape, indices, dtype):
        self.ndpointer, self.data_pointer = malloc_aligned(shape, dtype)
        self.ndpointer = Data(self.ndpointer,
                              [i.modulo if i.is_Stepping else None for i in indices])

    def __del__(self):
        free(self.data_pointer)
        self.data_pointer = None

    def fill(self, val):
        self.ndpointer.fill(val)


def malloc_aligned(shape, dtype=np.float32, alignment=None):
    """ Allocate memory using the C function malloc_aligned
    :param shape: Shape of the array to allocate
    :param dtype: Numpy datatype to allocate. Default to np.float32
    :param alignment: number of bytes to align to. Defaults to
    page size if not set.

    :returns (pointer, data_pointer) the first element of the tuple
    is the reference that can be used to access the data as a ctypes
    object. The second element is the low-level reference that is
    needed only for the call to free.
    """
    data_pointer = ctypes.cast(ctypes.c_void_p(), ctypes.POINTER(ctypes.c_float))
    arraysize = int(reduce(mul, shape))
    ctype = numpy_to_ctypes(dtype)
    if alignment is None:
        alignment = libc.getpagesize()

    ret = libc.posix_memalign(
        ctypes.byref(data_pointer),
        alignment,
        ctypes.c_ulong(arraysize * ctypes.sizeof(ctype))
    )
    if not ret == 0:
        error("Unable to allocate memory for shape %s", str(shape))
        return None

    data_pointer = ctypes.cast(
        data_pointer,
        np.ctypeslib.ndpointer(dtype=dtype, shape=shape)
    )

    pointer = np.ctypeslib.as_array(data_pointer, shape=shape)
    return (pointer, data_pointer)


def free(internal_pointer):
    """
    Use the C function free to free the memory allocated for the
    given pointer.
    """
    libc.free(internal_pointer)


def first_touch(array):
    """
    Uses an Operator to initialize the given array in the same pattern that
    would later be used to access it.
    """
    exp_init = [Eq(array.indexed[array.indices], 0)]
    op = devito.Operator(exp_init)
    op.apply()


class Data(np.ndarray):

    """
    A special :class:`numpy.ndarray` that supports logic indexing over modulo
    buffered dimensions.

    The type :class:`numpy.ndarray` is subclassed as indicated at: ::

        https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    The new instance takes an existing ndarray as input, casts it to one of
    type ``Data``, and adds the extra attribute ``modulo``.

    .. note::

        Any view or copy ``A`` created starting from ``self``, for instance via
        a slice operation or a universal function ("ufunc" in NumPy jargon), will
        still be of type :class:`Data`. However, if ``A`` is a view and its rank
        is lower than that of ``self``, namely ``A.ndim < self.ndim``, then the
        ``modulo`` attribute is dropped. A suitable exception is then raised if
        user code seems to attempt accessing data through modulo iteration on
        such a contracted view.
    """

    def __new__(cls, array, modulo=None):
        obj = np.asarray(array).view(cls)
        obj.modulo = tuple(modulo)
        return obj

    def __array_finalize__(self, obj):
        if type(obj) != Data:
            return
        # `self` is the newly created object
        # `obj` is the object from which `self` was created
        if self.ndim == obj.ndim:
            self.modulo = obj.modulo
        else:
            self.modulo = tuple(None for i in range(self.ndim))

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
            for i, j in zip(index, self.modulo):
                if j is None:
                    wrapped.append(i)
                elif isinstance(i, slice):
                    if i.start is None:
                        start = i.start
                    elif i.start >= 0:
                        start = i.start % j
                    else:
                        start = -(i.start % j)
                    if i.stop is None:
                        stop = i.stop
                    elif i.stop >= 0:
                        stop = i.stop % (j + 1)
                    else:
                        stop = -(i.stop % (j + 1))
                    wrapped.append(slice(start, stop, i.step))
                elif isinstance(i, (tuple, list)):
                    wrapped.append([k % j for k in i])
                else:
                    wrapped.append(i % j)
            return wrapped[0] if len(index) == 1 else tuple(wrapped)

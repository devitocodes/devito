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
    type ``Data``, and adds the extra attribute ``wrap``.
    """

    def __new__(cls, array, wrap=None):
        obj = np.asarray(array).view(cls)
        obj._wrap = wrap
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._wrap = getattr(obj, '_wrap', None)

    def __getitem__(self, index):
        index = self._apply_wrap(index)
        return super(Data, self).__getitem__(index)

    def __setitem__(self, index, val):
        index = self._apply_wrap(index)
        super(Data, self).__setitem__(index, val)

    def _apply_wrap(self, index):
        if isinstance(index, np.ndarray):
            # Mask array
            return index
        else:
            index = as_tuple(index)
            wrapped = []
            for i, j in zip(index, self._wrap):
                if j is None:
                    wrapped.append(i)
                elif isinstance(i, slice):
                    handle = []
                    handle.append(i.start if i.start is None else (i.start % j))
                    handle.append(i.stop if i.stop is None else (i.stop % (j + 1)))
                    handle.append(i.step)
                    wrapped.append(slice(*handle))
                elif isinstance(i, (tuple, list)):
                    wrapped.append([k % j for k in i])
                else:
                    wrapped.append(i % j)
            return wrapped[0] if len(index) == 1 else tuple(wrapped)

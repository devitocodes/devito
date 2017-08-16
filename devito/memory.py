from __future__ import absolute_import

import ctypes
from ctypes.util import find_library
from functools import reduce
from operator import mul

import numpy as np
from sympy import Eq

from devito.dimension import t
from devito.logger import error
from devito.tools import convert_dtype_to_ctype
import devito
"""
Pre-load ``libc`` to explicitly manage C memory
"""
libc = ctypes.CDLL(find_library('c'))


class CMemory(object):

    def __init__(self, shape, dtype=np.float32, alignment=None):
        self.ndpointer, self.data_pointer = malloc_aligned(shape, alignment, dtype)

    def __del__(self):
        free(self.data_pointer)
        self.data_pointer = None

    def fill(self, val):
        self.ndpointer.fill(val)


def malloc_aligned(shape, alignment=None, dtype=np.float32):
    """ Allocate memory using the C function malloc_aligned
    :param shape: Shape of the array to allocate
    :param alignment: number of bytes to align to. Defaults to
    page size if not set.
    :param dtype: Numpy datatype to allocate. Default to np.float32

    :returns (pointer, data_pointer) the first element of the tuple
    is the reference that can be used to access the data as a ctypes
    object. The second element is the low-level reference that is
    needed only for the call to free.
    """
    data_pointer = ctypes.cast(ctypes.c_void_p(), ctypes.POINTER(ctypes.c_float))
    arraysize = int(reduce(mul, shape))
    ctype = convert_dtype_to_ctype(dtype)
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
    """Use the C function free to free the memory allocated for the
    given pointer.
    """
    libc.free(internal_pointer)


def first_touch(array):
    """Uses the Propagator low-level API to initialize the given array(in Devito types)
    in the same pattern that would later be used to access it.
    """
    exp_init = [Eq(array.indexed[array.indices], 0)]
    op = devito.Operator(exp_init)
    op.apply()

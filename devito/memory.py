from __future__ import absolute_import

import ctypes
from ctypes.util import find_library
from functools import reduce
from operator import mul

import numpy as np
from sympy import Eq

from devito.logger import error
from devito.tools import convert_dtype_to_ctype


class CMemory(object):
    def __init__(self, shape, dtype=np.float32, alignment=None):
        self.ndpointer, self.data_pointer = malloc_aligned(shape, alignment, dtype)
        self.ndpointer.fill(0)

    def __del__(self):
        free(self.data_pointer)
        self.data_pointer = None


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
    libc = ctypes.CDLL(find_library('c'))
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
    libc = ctypes.CDLL(find_library('c'))
    libc.free(internal_pointer)


def first_touch(array):
    """Uses the Propagator low-level API to initialize the given array(in Devito types)
    in the same pattern that would later be used to access it.
    """
    from devito.propagator import Propagator
    from devito.interfaces import TimeData, PointData
    from devito.nodes import Iteration

    exp_init = [Eq(array.indexed[array.indices], 0)]
    it_init = []
    if isinstance(array, TimeData):
        shape = array.shape
        time_steps = shape[0]
        shape = shape[1:]
        space_dims = array.indices[1:]
    else:
        if isinstance(array, PointData):
            it_init = [Iteration(exp_init, dimension=array.indices[1],
                                 limits=array.shape[1])]
            exp_init = []
            time_steps = array.shape[0]
            shape = []
            space_dims = []
        else:
            shape = array.shape
            time_steps = 1
            space_dims = array.indices
    prop = Propagator(name="init", nt=time_steps, shape=shape,
                      stencils=exp_init, space_dims=space_dims)
    prop.add_devito_param(array)
    prop.save_vars[array.name] = True
    prop.time_loop_stencils_a = it_init
    prop.run([array.data])

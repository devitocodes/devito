from __future__ import absolute_import
import cgen
from devito.propagator import Propagator
import numpy as np
import ctypes
from ctypes.util import find_library
from operator import mul
from devito.tools import convert_dtype_to_ctype


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
    data_pointer = ctypes.c_void_p()
    arraysize = int(reduce(mul, shape))
    ctype = convert_dtype_to_ctype(dtype)
    if alignment is None:
        alignment = libc.getpagesize()
    libc.posix_memalign(
        ctypes.byref(data_pointer),
        alignment,
        arraysize * ctypes.sizeof(ctype)
    )
    data_pointer = ctypes.cast(
        data_pointer,
        np.ctypeslib.ndpointer(dtype=dtype, shape=shape)
    )
    pointer = np.ctypeslib.as_array(data_pointer, shape=shape)
    return (pointer, data_pointer)


def free(internal_pointer):
    """Use the C function free to free the memory allocated for the
    given pointer. Also sets the passed pointer to None.
    """
    libc = ctypes.CDLL(find_library('c'))
    libc.free(internal_pointer)
    internal_pointer = None


def first_touch(array):
    """Uses the Propagator low-level API to initialize the given array
    in the same pattern that would later be used to access it.
    """
    loop_body = cgen.Assign(array.c_element, 0)
    prop = Propagator(name="init", nt=1, shape=array.shape, loop_body=loop_body)
    prop.add_devito_param(array)
    prop.cfunction(array.data)

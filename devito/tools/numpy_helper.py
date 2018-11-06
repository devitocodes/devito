import ctypes
import numpy as np

from devito.tools.utils import prod

__all__ = ['numpy_to_ctypes', 'numpy_to_mpitypes', 'numpy_view_offsets']


def numpy_to_ctypes(dtype):
    """Map numpy types to ctypes types."""
    return {np.int32: ctypes.c_int,
            np.float32: ctypes.c_float,
            np.int64: ctypes.c_int64,
            np.float64: ctypes.c_double}[dtype]


def numpy_to_mpitypes(dtype):
    """Map numpy types to MPI datatypes."""
    return {np.int32: 'MPI_INT',
            np.float32: 'MPI_FLOAT',
            np.int64: 'MPI_LONG',
            np.float64: 'MPI_DOUBLE'}[dtype]


def numpy_view_offsets(array, base=None):
    """
    Retrieve the offset of a view from its base array along each dimension and side.

    :param array: A :class:`numpy.ndarray`.
    :param base: The base of ``array``. Most of the times the ``base`` is available
                 through ``array.base``. However, if this function is to be called
                 within ``__array_finalize__``, where ``base`` hasn't been set yet,
                 the ``base`` has to be provided explicitly
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Expected a `numpy.ndarray`, got `%s`" % type(array))
    if array.base is None:
        if base is None:
            raise ValueError("Cannot access ``array``'s base.")
    else:
        base = array.base

    start_byte_distance = np.byte_bounds(array)[0] - np.byte_bounds(base)[0]
    start_elem_distance = start_byte_distance // array.itemsize
    assert start_byte_distance % array.itemsize == 0

    end_byte_distance = np.byte_bounds(array)[1] - np.byte_bounds(base)[0]
    end_elem_distance = (end_byte_distance // array.itemsize) - 1
    assert end_byte_distance % array.itemsize == 0

    offsets = []
    for i, s in enumerate(base.shape):
        hyperplane_size = prod(base.shape[i+1:])

        # Start
        lofs = start_elem_distance // hyperplane_size
        start_elem_distance -= lofs*hyperplane_size

        # End
        rofs = end_elem_distance // hyperplane_size
        end_elem_distance -= rofs*hyperplane_size

        offsets.append((lofs, s-rofs-1))

    return tuple(offsets)

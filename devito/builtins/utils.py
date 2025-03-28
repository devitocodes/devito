from functools import wraps

import numpy as np

import devito as dv
from devito.arch import Device
from devito.symbolics import uxreplace
from devito.tools import as_tuple

__all__ = ['make_retval', 'nbl_to_padsize', 'pad_outhalo', 'abstract_args',
           'check_builtins_args']


accumulator_mapper = {
    # Integer accumulates on Float64
    np.int8: np.float64, np.uint8: np.float64,
    np.int16: np.float64, np.uint16: np.float64,
    np.int32: np.float64, np.uint32: np.float64,
    np.int64: np.float64, np.uint64: np.float64,
    # FloatX accumulates on Float2X
    np.float16: np.float32,
    np.float32: np.float64,
    # NOTE: np.float128 isn't really a thing, see for example
    # https://github.com/numpy/numpy/issues/10288
    # https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1070
    np.float64: np.float64,
    # ComplexX accumulates on Complex2X
    np.complex64: np.complex128,
    np.complex128: np.complex128,
}


def make_retval(f):
    """
    Devito does not support passing values by reference. This function
    creates a dummy Function of size 1 to store the return value of a builtin
    applied to `f`.
    """
    if f.grid is None:
        raise ValueError("No Grid available")

    cls = make_retval.cls or dv.Function

    dtype = accumulator_mapper[f.dtype]

    i = dv.Dimension(name='mri',)
    n = cls(name='n', shape=(1,), dimensions=(i,), grid=f.grid,
            dtype=dtype, space='host')

    n.data[:] = 0

    return n
make_retval.cls = None  # noqa


def nbl_to_padsize(nbl, ndim):
    """
    Creates the pad sizes from `nbl`. The output is a tuple of tuple
    such as ((40, 40), (40, 40)) where each subtuple is the left/right
    pad size for the dimension.
    """
    nb_total = as_tuple(nbl)
    if len(nb_total) == 1 and len(nb_total) < ndim:
        nb_pad = ndim*((nb_total[0], nb_total[0]),)
        slices = ndim*(slice(nb_total[0], None, 1),) \
            if nb_total[0] == 0 else ndim*(slice(nb_total[0], -nb_total[0], 1),)
    elif len(nb_total) == ndim:
        nb_pad = []
        slices = []
        for nb_dim in nb_total:
            if len(as_tuple(nb_dim)) == 1:
                nb_pad.append((nb_dim, nb_dim))
                s = slice(nb_dim, None, 1) if nb_dim == 0 \
                    else slice(nb_dim, -nb_dim, 1)
                slices.append(s)
            else:
                assert len(nb_dim) == 2
                nb_pad.append(nb_dim)
                s = slice(nb_dim[0], None, 1) if nb_dim[1] == 0 \
                    else slice(nb_dim[0], -nb_dim[1], 1)
                slices.append(s)
    else:
        raise ValueError("`nbl` must be an integer or tuple of (tuple of) integers"
                         "of length `function.ndim`.")
    return tuple(nb_pad), tuple(slices)


def pad_outhalo(function):
    """ Pad outer halo with edge values."""
    h_shape = function._data_with_outhalo.shape
    for i, h in enumerate(function._size_outhalo):
        slices = [slice(None)]*len(function.shape)
        slices_d = [slice(None)]*len(function.shape)
        if h.left != 0:
            # left part
            slices[i] = slice(0, h.left)
            slices_d[i] = slice(h.left, h.left+1, 1)
            function._data_with_outhalo._local[tuple(slices)] \
                = function._data_with_outhalo._local[tuple(slices_d)]
        if h.right != 0:
            # right part
            slices[i] = slice(h_shape[i] - h.right, h_shape[i], 1)
            slices_d[i] = slice(h_shape[i] - h.right - 1, h_shape[i] - h.right, 1)
            function._data_with_outhalo._local[tuple(slices)] \
                = function._data_with_outhalo._local[tuple(slices_d)]
        if h.left == 0 and h.right == 0:
            # Need to access it so that that worker is not blocking exectution since
            # _data_with_outhalo requires communication
            function._data_with_outhalo._local[0] = function._data_with_outhalo._local[0]


def abstract_args(func):
    """
    Turn user-provided arguments into abstract parameters to construct generic
    Operators. This minimizes the number of constructed and jit-compiled builtins.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        abstract_objects = dv.passes.iet.engine.abstract_objects
        mapper = abstract_objects(args)

        processed = []
        argmap = {}
        for a in args:
            try:
                if a.is_DiscreteFunction:
                    v = uxreplace(a, mapper)
                    processed.append(v)
                    argmap[v.name] = a
                    continue
            except AttributeError:
                pass
            processed.append(a)

        return func(*processed, argmap=argmap, **kwargs)

    return wrapper


def check_builtins_args(func):
    """
    Perform checks on the arguments supplied to a builtin.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        platform = dv.configuration['platform']
        if not isinstance(platform, Device):
            return func(*args, **kwargs)

        for i in args:
            try:
                if not i.is_persistent:
                    raise ValueError(f"Cannot apply `{func.__name__}` to transient "
                                     f"function `{i.name}` on backend `{platform}`")
            except AttributeError:
                pass

        return func(*args, **kwargs)

    return wrapper

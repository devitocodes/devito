from functools import wraps

import numpy as np

import devito as dv
from devito.symbolics import uxreplace
from devito.tools import as_tuple

__all__ = ['MPIReduction', 'nbl_to_padsize', 'pad_outhalo', 'abstract_args']


class MPIReduction(object):
    """
    A context manager to build MPI-aware reduction Operators.
    """

    def __init__(self, *functions, op=dv.mpi.MPI.SUM, dtype=None):
        grids = {f.grid for f in functions}
        if len(grids) == 0:
            self.grid = None
        elif len(grids) == 1:
            self.grid = grids.pop()
        else:
            raise ValueError("Multiple Grids found")
        if dtype is not None:
            self.dtype = dtype
        else:
            dtype = {f.dtype for f in functions}
            if len(dtype) == 1:
                self.dtype = dtype.pop()
            else:
                raise ValueError("Illegal mixed data types")
        self.v = None
        self.op = op

    def __enter__(self):
        i = dv.Dimension(name='i',)
        self.n = dv.Function(name='n', shape=(1,), dimensions=(i,),
                             grid=self.grid, dtype=self.dtype)
        self.n.data[0] = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.grid is None or not dv.configuration['mpi']:
            assert self.n.data.size == 1
            self.v = self.n.data[0]
        else:
            comm = self.grid.distributor.comm
            self.v = comm.allreduce(np.asarray(self.n.data), self.op)[0]


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

import numpy as np

import devito as dv
from devito.builtins.utils import MPIReduction


__all__ = ['norm', 'sumall', 'inner', 'mmin', 'mmax']

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
    np.float64: np.float64
}


@dv.switchconfig(log_level='ERROR')
def norm(f, order=2):
    """
    Compute the norm of a Function.

    Parameters
    ----------
    f : Function
        Input Function.
    order : int, optional
        The order of the norm. Defaults to 2.
    """
    Pow = dv.finite_differences.differentiable.Pow
    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    dtype = accumulator_mapper[f.dtype]
    s = dv.types.Symbol(name='sum', dtype=dtype)

    with MPIReduction(f, dtype=dtype) as mr:
        op = dv.Operator([dv.Eq(s, 0.0)] +
                         eqns +
                         [dv.Inc(s, dv.Abs(Pow(p, order))), dv.Eq(mr.n[0], s)],
                         name='norm%d' % order)
        op.apply(**kwargs)

    v = np.power(mr.v, 1/order)

    return f.dtype(v)


@dv.switchconfig(log_level='ERROR')
def sumall(f):
    """
    Compute the sum of all Function data.

    Parameters
    ----------
    f : Function
        Input Function.
    """
    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    dtype = accumulator_mapper[f.dtype]
    s = dv.types.Symbol(name='sum', dtype=dtype)

    with MPIReduction(f, dtype=dtype) as mr:
        op = dv.Operator([dv.Eq(s, 0.0)] +
                         eqns +
                         [dv.Inc(s, p), dv.Eq(mr.n[0], s)],
                         name='sum')
        op.apply(**kwargs)

    return f.dtype(mr.v)


@dv.switchconfig(log_level='ERROR')
def inner(f, g):
    """
    Inner product of two Functions.

    Parameters
    ----------
    f : Function
        First input operand
    g : Function
        Second input operand

    Raises
    ------
    ValueError
        If the two input Functions are defined over different grids, or have
        different dimensionality, or their dimension-wise sizes don't match.
        If in input are two SparseFunctions and their coordinates don't match,
        the exception is raised.

    Notes
    -----
    The inner product is the sum of all dimension-wise products. For 1D Functions,
    the inner product corresponds to the dot product.
    """
    # Input check
    if f.is_TimeFunction and f._time_buffering != g._time_buffering:
        raise ValueError("Cannot compute `inner` between save/nosave TimeFunctions")
    if f.shape != g.shape:
        raise ValueError("`f` and `g` must have same shape")
    if f._data is None or g._data is None:
        raise ValueError("Uninitialized input")
    if f.is_SparseFunction and not np.all(f.coordinates_data == g.coordinates_data):
        raise ValueError("Non-matching coordinates")

    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    rhs, eqns = f.guard(f*g) if f.is_SparseFunction else (f*g, [])

    dtype = accumulator_mapper[f.dtype]
    s = dv.types.Symbol(name='sum', dtype=dtype)

    with MPIReduction(f, g, dtype=dtype) as mr:
        op = dv.Operator([dv.Eq(s, 0.0)] +
                         eqns +
                         [dv.Inc(s, rhs), dv.Eq(mr.n[0], s)],
                         name='inner')
        op.apply(**kwargs)

    return f.dtype(mr.v)


@dv.switchconfig(log_level='ERROR')
def mmin(f):
    """
    Retrieve the minimum.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """
    if isinstance(f, dv.Constant):
        return f.data
    elif isinstance(f, dv.types.dense.DiscreteFunction):
        with MPIReduction(f, op=dv.mpi.MPI.MIN) as mr:
            mr.n.data[0] = np.min(f.data_ro_domain).item()
        return mr.v.item()
    else:
        raise ValueError("Expected Function, not `%s`" % type(f))


@dv.switchconfig(log_level='ERROR')
def mmax(f):
    """
    Retrieve the maximum.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """
    if isinstance(f, dv.Constant):
        return f.data
    elif isinstance(f, dv.types.dense.DiscreteFunction):
        with MPIReduction(f, op=dv.mpi.MPI.MAX) as mr:
            mr.n.data[0] = np.max(f.data_ro_domain).item()
        return mr.v.item()
    else:
        raise ValueError("Expected Function, not `%s`" % type(f))

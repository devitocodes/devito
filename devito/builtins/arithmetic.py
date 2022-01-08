import numpy as np

import devito as dv
from devito.builtins.utils import MPIReduction


__all__ = ['norm', 'sumall', 'inner', 'mmin', 'mmax', 'count_nonzero', 'nonzero']


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

    s = dv.types.Symbol(name='sum', dtype=f.dtype)

    with MPIReduction(f) as mr:
        op = dv.Operator([dv.Eq(s, 0.0)] +
                         eqns +
                         [dv.Inc(s, dv.Abs(Pow(p, order))), dv.Eq(mr.n[0], s)],
                         name='norm%d' % order)
        op.apply(**kwargs)

    v = np.power(mr.v, 1/order)

    return f.dtype(v)


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

    s = dv.types.Symbol(name='sum', dtype=f.dtype)

    with MPIReduction(f) as mr:
        op = dv.Operator([dv.Eq(s, 0.0)] +
                         eqns +
                         [dv.Inc(s, p), dv.Eq(mr.n[0], s)],
                         name='sum')
        op.apply(**kwargs)

    return f.dtype(mr.v)


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

    s = dv.types.Symbol(name='sum', dtype=f.dtype)

    with MPIReduction(f, g) as mr:
        op = dv.Operator([dv.Eq(s, 0.0)] +
                         eqns +
                         [dv.Inc(s, rhs), dv.Eq(mr.n[0], s)],
                         name='inner')
        op.apply(**kwargs)

    return f.dtype(mr.v)


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


def count_nonzero(f):
    """
    Retrieve the count of nonzero elements of a (Time)Function.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """
    ci = dv.ConditionalDimension(name='ci', parent=f.dimensions[-1],
                                 condition=dv.Ne(f, 0))
    g = dv.Function(name='g', shape=(1,), dimensions=(ci,), space_order=0, dtype=np.int32)

    # Extra Scalar used to avoid reduction in gcc-5
    i = dv.types.Scalar(name='i', dtype=np.int32)
    eqi = dv.Eq(i, 0)

    eq0 = dv.Inc(g[i], 1, implicit_dims=(f.dimensions + (ci,)))
    op0 = dv.Operator([eqi, eq0], opt=('advanced', {'par-collapse-ncores': 1000}))
    op0.apply()

    return g.data[0]


def nonzero(f):
    """
    Retrieve the count of nonzero elements of a (Time)Function.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """

    # Get the nonzero count to allocate only the necessary space for nonzero elements
    count = count_nonzero(f)

    # Dimension used only to nest different size of Functions under the same dim
    id_dim = dv.Dimension(name='id_dim')

    # Conditional for nonzero element
    ci = dv.ConditionalDimension(name='ci', parent=f.dimensions[-1],
                                 condition=dv.Ne(f, 0))
    g = dv.Function(name='g', shape=(count, len(f.dimensions)),
                    dimensions=(ci, id_dim), dtype=np.int32, space_order=0)

    eqs = []

    # Extra Scalar used to avoid reduction in gcc-5
    k = dv.types.Scalar(name='k', dtype=np.int32)
    eqi = dv.Eq(k, -1)
    eqs.append(eqi)
    eqii = dv.Inc(k, 1, implicit_dims=(f.dimensions + (ci,)))
    eqs.append(eqii)

    for n, i in enumerate(f.dimensions):
        eqs.append(dv.Eq(g[k, n], f.dimensions[n], implicit_dims=(f.dimensions + (ci,))))

    op0 = dv.Operator(eqs)
    op0.apply()
    return g.data[:]

"""
Built-in :class:`Operator`s provided by Devito.
"""

from sympy import Abs, Pow
import numpy as np

import devito as dv


def assign(f, v=0):
    """
    Assign a value to a :class:`Function`.

    Parameters
    ----------
    f : Function
        The left-hand side of the assignment.
    v : scalar, optional
        The right-hand side of the assignment.
    """
    dv.Operator(dv.Eq(f, v), name='assign')()


def smooth(f, g, axis=None):
    """
    Smooth a :class:`Function` through simple moving average.

    Parameters
    ----------
    f : Function
        The left-hand side of the smoothing kernel, that is the smoothed Function.
    g : Function
        The right-hand side of the smoothing kernel, that is the Function being smoothed.
    axis : Dimension or list of Dimensions, optional
        The :class:`Dimension` along which the smoothing operation is performed.
        Defaults to ``f``'s innermost Dimension.

    Notes
    -----
    More info about simple moving average available at: ::

        https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    """
    if g.is_Constant:
        # Return a scaled version of the input if it's a Constant
        f.data[:] = .9 * g.data
    else:
        if axis is None:
            axis = g.dimensions[-1]
        dv.Operator(dv.Eq(f, g.avg(dims=axis)), name='smoother')()


def norm(f, order=2):
    """
    Compute the norm of a :class:`Function`.

    Parameters
    ----------
    f : Function
        Input Function.
    order : int, optional
        The order of the norm. Defaults to 2.
    """
    i = dv.Dimension(name='i',)
    n = dv.Function(name='n', shape=(1,), dimensions=(i,), grid=f.grid, dtype=f.dtype)
    n.data[0] = 0

    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    op = dv.Operator(eqns + [dv.Inc(n[0], Abs(Pow(p, order)))], name='norm%d' % order)
    op.apply(**kwargs)

    # May need a global reduction over MPI
    if f.grid is None:
        assert n.data.size == 1
        v = n.data[0]
    else:
        comm = f.grid.distributor.comm
        v = comm.allreduce(np.asarray(n.data))[0]

    v = Pow(v, 1/order)

    return np.float(v)


def sumall(f):
    """
    Compute the sum of the values in a :class:`Function`.

    Parameters
    ----------
    f : Function
        Input Function.
    """
    i = dv.Dimension(name='i',)
    n = dv.Function(name='n', shape=(1,), dimensions=(i,), grid=f.grid, dtype=f.dtype)
    n.data[0] = 0

    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    op = dv.Operator(eqns + [dv.Inc(n[0], p)], name='sum')
    op.apply(**kwargs)

    # May need a global reduction over MPI
    if f.grid is None:
        assert n.data.size == 1
        v = n.data[0]
    else:
        comm = f.grid.distributor.comm
        v = comm.allreduce(np.asarray(n.data))[0]

    return np.float(v)

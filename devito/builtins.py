"""
Built-in :class:`Operator`s provided by Devito.
"""

from sympy import Abs, sqrt

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
        The Function for which the norm is computed.
    order : int, optional
        The order of the norm. Defaults to 2.
    """
    n = dv.Constant(name='n', dtype=f.dtype)
    if order == 1:
        dv.Operator(dv.Inc(n, Abs(f)), name='norm1')()
    elif order == 2:
        dv.Operator([dv.Eq(n, f*f), dv.Eq(n, sqrt(n))], name='norm2')()
    else:
        raise NotImplementedError
    return n.data

from sympy import exp, Min
import numpy as np
from devito import Eq, Operator

__all__ = ['setup_w_over_q']


def setup_w_over_q(wOverQ, w, qmin, qmax, npad, sigma=0):
    """
    Initialise spatially variable w/Q field used to implement attenuation and
    absorb outgoing waves at the edges of the model. Uses Devito Operator.

    Parameters
    ----------
    wOverQ : Function, required
        The omega over Q field used to implement attenuation in the model,
        and the absorbing boundary condition for outgoing waves.
    w : float32, required
        center angular frequency, e.g. peak frequency of Ricker source wavelet
        used for modeling.
    qmin : float32, required
        Q value at the edge of the model. Typically set to 0.1 to strongly
        attenuate outgoing waves.
    qmax : float32, required
        Q value in the interior of the model. Typically set to 100 as a
        reasonable and physically meaningful Q value.
    npad : int, required
        Number of points in the absorbing boundary region. Note that we expect
        this to be the same on all sides of the model.
    sigma : float32, optional, defaults to None
        sigma value for call to scipy gaussian smoother, default 5.
    """
    # sanity checks
    assert w > 0, "supplied w value [%f] must be positive" % (w)
    assert qmin > 0, "supplied qmin value [%f] must be positive" % (qmin)
    assert qmax > 0, "supplied qmax value [%f] must be positive" % (qmax)
    assert npad > 0, "supplied npad value [%f] must be positive" % (npad)
    for n in wOverQ.grid.shape:
        if n - 2*npad < 1:
            raise ValueError("2 * npad must not exceed dimension size!")

    lqmin = np.log(qmin)
    lqmax = np.log(qmax)

    # 1. Get distance to closest boundary in all dimensions
    # 2. Logarithmic variation between qmin, qmax across the absorbing boundary
    pos = Min(1, Min(*[Min(d - d.symbolic_min, d.symbolic_max - d)
                       for d in wOverQ.dimensions]) / npad)
    val = exp(lqmin + pos * (lqmax - lqmin))

    # 2020.05.04 currently does not support spatial smoothing of the Q field
    # due to MPI weirdness in reassignment of the numpy array
    eqn1 = Eq(wOverQ, w / val)
    Operator([eqn1], name='WOverQ_Operator')()

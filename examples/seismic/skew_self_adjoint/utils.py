from sympy import exp, Min
import numpy as np
from devito import (Grid, Constant, Function, SpaceDimension, Eq, Operator)
from examples.seismic import TimeAxis
# from devito.builtins import gaussian_smooth

__all__ = ['compute_critical_dt', 'setup_w_over_q', 'default_setup_iso']


def compute_critical_dt(v):
    """
    Determine the temporal sampling to satisfy CFL stability.
    This method replicates the functionality in the Model class.

    Note we add a safety factor, reducing dt by a factor 0.75 due to the
    w/Q attentuation term.

    Parameters
    ----------
    v : Function
        velocity
    """
    coeff = 0.38 if len(v.grid.shape) == 3 else 0.42
    dt = 0.75 * v.dtype(coeff * np.min(v.grid.spacing) / (np.max(v.data)))
    return v.dtype("%.5e" % dt)


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

#     eqn1 = Eq(wOverQ, val)
#     Operator([eqn1], name='WOverQ_Operator_init')()
#     # If we apply the smoother, we must renormalize output to [qmin,qmax]
#     if sigma > 0:
#         smooth = gaussian_smooth(wOverQ.data, sigma=sigma)
#         smin, smax = np.min(smooth), np.max(smooth)
#         smooth[:] = qmin + (qmax - qmin) * (smooth - smin) / (smax - smin)
#         wOverQ.data[:] = smooth
#     eqn2 = Eq(wOverQ, w / wOverQ)
#     Operator([eqn2], name='WOverQ_Operator_recip')()


def default_setup_iso(npad, shape, dtype,
                      sigma=0, qmin=0.1, qmax=100.0, tmin=0.0, tmax=2000.0,
                      bvalue=1.0/1000.0, vvalue=1.5, space_order=8):
    """
    For isotropic propagator build default model with 10m spacing,
        and 1.5 m/msec velocity

    Return:
        dictionary of velocity, buoyancy, and wOverQ
        TimeAxis defining temporal sampling
        Source locations: one located at center of model, z = 1dz
        Receiver locations, one per interior grid intersection, z = 2dz
            2D: 1D grid of receivers center z covering interior of model
            3D: 2D grid of receivers center z covering interior of model
    """
    d = 10.0
    origin = tuple([0.0 - d * npad for s in shape])
    extent = tuple([d * (s - 1) for s in shape])

    # Define dimensions
    if len(shape) == 2:
        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=d))
        z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=d))
        grid = Grid(extent=extent, shape=shape, origin=origin,
                    dimensions=(x, z), dtype=dtype)
    else:
        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=d))
        y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=d))
        z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=d))
        grid = Grid(extent=extent, shape=shape, origin=origin,
                    dimensions=(x, y, z), dtype=dtype)

    b = Function(name='b', grid=grid, space_order=space_order)
    v = Function(name='v', grid=grid, space_order=space_order)
    b.data[:] = bvalue
    v.data[:] = vvalue

    dt = dtype("%.6f" % (0.8 * compute_critical_dt(v)))
    time_axis = TimeAxis(start=tmin, stop=tmax, step=dt)

    # Define coordinates in 2D and 3D
    if len(shape) == 2:
        nr = shape[0] - 2 * npad
        src_coords = np.empty((1, len(shape)), dtype=dtype)
        rec_coords = np.empty((nr, len(shape)), dtype=dtype)
        src_coords[:, 0] = origin[0] + extent[0] / 2
        src_coords[:, 1] = 1 * d
        rec_coords[:, 0] = np.linspace(0.0, d * (nr - 1), nr)
        rec_coords[:, 1] = 2 * d
    else:
        # using numpy outer product here for array iteration speed
        xx, yy = np.ogrid[:shape[0]-2*npad, :shape[1]-2*npad]
        x1 = np.ones((shape[0] - 2 * npad, 1))
        y1 = np.ones((1, shape[1] - 2 * npad))
        xcoord = (xx*y1).reshape(-1)
        ycoord = (x1*yy).reshape(-1)
        nr = len(xcoord)
        src_coords = np.empty((1, len(shape)), dtype=dtype)
        rec_coords = np.empty((nr, len(shape)), dtype=dtype)
        src_coords[:, 0] = origin[0] + extent[0] / 2
        src_coords[:, 1] = origin[1] + extent[1] / 2
        src_coords[:, 2] = 1 * d
        rec_coords[:, 0] = d * xcoord
        rec_coords[:, 1] = d * ycoord
        rec_coords[:, 2] = 2 * d

    return b, v, time_axis, src_coords, rec_coords

import numpy as np

from examples.containers import IShot
from examples.tti.TTI_codegen import TTI_cg
from examples.seismic import Model
from devito.logger import warning


def source(t, f0):
    agauss = 0.5*f0
    tcut = 1.5/agauss
    s = (t-tcut)*agauss
    return np.exp(-2*s**2)*np.cos(2*np.pi*s)


def setup(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
          time_order=2, space_order=4, nbpml=10):

    origin = (0., 0., 0.)

    # True velocity
    true_vp = np.ones(dimensions) + 1.0
    true_vp[:, :, int(dimensions[0] / 3):int(2*dimensions[0]/3)] = 3.0
    true_vp[:, :, int(2*dimensions[0] / 3):int(dimensions[0])] = 4.0

    model = Model(origin, spacing, dimensions, true_vp,
                  nbpml=nbpml,
                  epsilon=.4*np.ones(dimensions),
                  delta=-.1*np.ones(dimensions),
                  theta=-np.pi/7*np.ones(dimensions),
                  phi=np.pi/5*np.ones(dimensions))

    # Define seismic data.
    data = IShot()
    src = IShot()

    f0 = .010
    dt = model.critical_dt
    t0 = 0.0
    nt = int(1+(tn-t0)/dt)
    # Set up the source as Ricker wavelet for f0

    # Source geometry
    time_series = np.zeros((nt, 1))

    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

    location = np.zeros((1, 3))
    location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
    location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    location[0, 2] = origin[1] + 2 * spacing[2]
    src.set_receiver_pos(location)
    src.set_shape(nt, 1)
    src.set_traces(time_series)

    # Receiver geometry
    receiver_coords = np.zeros((101, 3))
    receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                        dimensions[0] * spacing[0], num=101)
    receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    receiver_coords[:, 2] = location[0, 1]
    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, 101)

    return TTI_cg(model, data, src, t_order=time_order, s_order=space_order)


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        time_order=2, space_order=4, nbpml=10, **kwargs):

    TTI = setup(dimensions, spacing, tn, time_order, space_order, nbpml)

    if space_order % 4 != 0:
        warning('WARNING: TTI requires a space_order that is a multiple of 4!')

    rec, u, v, gflopss, oi, timings = TTI.Forward(**kwargs)

    return gflopss, oi, timings, [rec, u, v]


if __name__ == "__main__":
    run(autotune=True)

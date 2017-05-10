import numpy as np

from devito.logger import info
from examples.acoustic import AcousticWaveSolver
from examples.seismic import Model, PointSource, Receiver


# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape)
    out[:, :] = vel[:, :]
    nx = shape[0]

    for a in range(5, nx-6):
        out[a, :] = np.sum(vel[a - 5:a + 5, :], axis=0) / 10

    return out


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))

    return (1-2.*r**2)*np.exp(-r**2)


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        time_order=2, space_order=4, nbpml=40, full_run=False, **kwargs):

    origin = (0., 0., 0.)

    # True velocity
    true_vp = np.ones(dimensions) + .5
    if len(dimensions) == 2:
        true_vp[:, int(dimensions[0] / 2):dimensions[0]] = 2.5
    else:
        true_vp[:, :, int(dimensions[0] / 2):dimensions[0]] = 2.5

    # Smooth velocity
    initial_vp = smooth10(true_vp, dimensions)

    dm = 1. / (true_vp * true_vp) - 1. / (initial_vp * initial_vp)

    model = Model(origin, spacing, dimensions, true_vp, nbpml=nbpml)

    f0 = .010
    dt = model.critical_dt
    if time_order == 4:
        dt *= 1.73
    t0 = 0.0
    nt = int(1+(tn-t0)/dt)

    # Source geometry
    time_series = np.zeros((nt, 1))
    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

    location = np.zeros((1, 3))
    location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
    location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    location[0, 2] = origin[1] + 2 * spacing[2]
    src = PointSource(name='src', data=time_series, coordinates=location)

    # Receiver geometry
    receiver_coords = np.zeros((101, 3))
    receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                        dimensions[0] * spacing[0], num=101)
    receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    receiver_coords[:, 2] = location[0, 1]
    rec = Receiver(name='rec', ntime=nt, coordinates=receiver_coords)

    # Create wave solver from model, source and receiver definitions
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                time_order=time_order,
                                space_order=space_order, **kwargs)

    info("Applying Forward")
    rec, u, summary = solver.Forward(save=full_run, **kwargs)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u.data]

    info("Applying Adjoint")
    solver.Adjoint(rec, **kwargs)
    info("Applying Born")
    solver.Born(dm, **kwargs)
    info("Applying Gradient")
    solver.Gradient(rec, u, **kwargs)


if __name__ == "__main__":
    run(full_run=True, autotune=False, space_order=6, time_order=2)

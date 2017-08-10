import numpy as np

from devito.logger import info
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Model, RickerSource, Receiver


# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape, dtype=np.float32)
    out[:] = vel[:]
    nz = shape[-1]

    for a in range(5, nz-6):
        if len(shape) == 2:
            out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
        else:
            out[:, :, a] = np.sum(vel[:, :, a - 5:a + 5], axis=2) / 10

    return out


def acoustic_setup(dimensions=(50, 50, 50), spacing=(15.0, 15.0, 15.0),
                   tn=500., time_order=2, space_order=4, nbpml=10, **kwargs):

    ndim = len(dimensions)
    nrec = dimensions[0]
    origin = tuple([0.]*ndim)
    spacing = spacing[:ndim]

    # Velocity model, two layers
    true_vp = np.ones(dimensions) + .5
    true_vp[..., int(dimensions[-1] / 3):dimensions[-1]] = 2.5

    # Define seismic data
    model = Model(origin, spacing, dimensions, true_vp, nbpml=int(nbpml))

    # Derive timestepping from model spacing
    dt = model.critical_dt
    if time_order == 4:
        dt *= 1.73
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)
    time = np.linspace(t0, tn, nt)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', ndim=ndim, f0=0.01, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, lust below surface)
    rec = Receiver(name='nrec', ntime=nt, npoint=nrec, ndim=ndim)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                time_order=time_order,
                                space_order=space_order, **kwargs)
    return solver


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        time_order=2, space_order=4, nbpml=40, full_run=False, **kwargs):

    solver = acoustic_setup(dimensions=dimensions, spacing=spacing,
                            nbpml=nbpml, tn=tn, space_order=space_order,
                            time_order=time_order, **kwargs)

    initial_vp = smooth10(solver.model.m.data, solver.model.shape_domain)
    dm = np.float32(initial_vp**2 - solver.model.m.data)
    info("Applying Forward")
    rec, u, summary = solver.forward(save=full_run)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u.data]

    info("Applying Adjoint")
    solver.adjoint(rec, **kwargs)
    info("Applying Born")
    solver.born(dm, **kwargs)
    info("Applying Gradient")
    solver.gradient(rec, u, **kwargs)


if __name__ == "__main__":
    run(full_run=True, autotune=False, space_order=6, time_order=2)

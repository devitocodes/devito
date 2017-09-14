import numpy as np
from argparse import ArgumentParser

from devito.logger import info
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, RickerSource, Receiver


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
    nrec = dimensions[0]
    model = demo_model('layers', ratio=3, shape=dimensions,
                       spacing=spacing, nbpml=nbpml)

    # Derive timestepping from model spacing
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time = np.linspace(t0, tn, nt)  # Discretized time axis

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', ndim=model.dim, f0=0.01, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, just below surface)
    rec = Receiver(name='nrec', ntime=nt, npoint=nrec, ndim=model.dim)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                time_order=time_order,
                                space_order=space_order, **kwargs)
    return solver


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        time_order=2, space_order=4, nbpml=40, full_run=False,
        autotune=False, **kwargs):

    solver = acoustic_setup(dimensions=dimensions, spacing=spacing,
                            nbpml=nbpml, tn=tn, space_order=space_order,
                            time_order=time_order, **kwargs)

    initial_vp = smooth10(solver.model.m.data, solver.model.shape_domain)
    dm = np.float32(initial_vp**2 - solver.model.m.data)
    info("Applying Forward")
    rec, u, summary = solver.forward(save=full_run, autotune=autotune, **kwargs)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u.data]

    info("Applying Adjoint")
    solver.adjoint(rec, autotune=autotune, **kwargs)
    info("Applying Born")
    solver.born(dm, autotune=autotune, **kwargs)
    info("Applying Gradient")
    solver.gradient(rec, u, autotune=autotune, **kwargs)


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-to", "--time_order", default=2,
                        type=int, help="Time order of the simulation")
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    args = parser.parse_args()

    run(full_run=args.full, autotune=args.autotune,
        space_order=args.space_order, time_order=args.time_order,
        nbpml=args.nbpml)

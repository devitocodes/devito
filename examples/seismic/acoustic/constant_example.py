import numpy as np
from argparse import ArgumentParser

from devito.logger import info
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Model, RickerSource, Receiver
from devito import Constant


def acoustic_setup(dimensions=(50, 50, 50), spacing=(15.0, 15.0, 15.0), tn=500.,
                   time_order=2, space_order=4, nbpml=10, vp=1.5, **kwargs):

    ndim = len(dimensions)
    nrec = dimensions[0]
    origin = tuple([0.]*ndim)
    spacing = spacing[:ndim]

    # Velocity model, constant
    true_vp = vp

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

    # Note: A potential bug in GCC requires using less aggressive loop
    # transformations. In particular, the DLE `_avoid_denormals` pass is skipped
    # by using `advanced-safemath`, rather than the default DLE mode `advanced`.

    return AcousticWaveSolver(model, source=src, receiver=rec, dle='advanced-safemath',
                              time_order=time_order, space_order=space_order, **kwargs)


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        time_order=2, space_order=4, nbpml=40, full_run=False, **kwargs):

    solver = acoustic_setup(dimensions=dimensions, spacing=spacing, nbpml=nbpml,
                            tn=tn, space_order=space_order,
                            time_order=time_order, **kwargs)

    initial_vp = 1.8
    dm = (initial_vp**2 - solver.model.m.data) * np.ones(solver.model.shape_domain,
                                                         dtype=np.float32)
    info("Applying Forward")
    # Default model.m
    rec, u, summary = solver.forward(save=full_run)
    # With  a new m as Constant
    m0 = Constant(name="m", value=.25, dtype=np.float32)
    solver.forward(save=full_run, m=m0)
    # With a new m as a scalar value
    solver.forward(save=full_run, m=.25)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u.data]

    info("Applying Adjoint")
    solver.adjoint(rec, **kwargs)
    info("Applying Born")
    solver.born(dm, **kwargs)
    info("Applying Gradient")
    solver.gradient(rec, u, **kwargs)


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
    parser.add_argument("--nbpml", default=0,
                        type=int, help="Number of PML layers around the domain")
    args = parser.parse_args()

    run(full_run=args.full, autotune=args.autotune,
        space_order=args.space_order, time_order=args.time_order,
        dimensions=(50, 50), tn=500., nbpml=args.nbpml)

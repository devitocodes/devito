import numpy as np
from argparse import ArgumentParser

from devito.logger import warning
from examples.seismic import demo_model, Receiver, RickerSource
from examples.seismic.tti import AnisotropicWaveSolver


def tti_setup(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
              time_order=2, space_order=4, nbpml=10, **kwargs):

    nrec = 101
    # Two layer model for true velocity
    model = demo_model('constanttti', shape=dimensions, nbpml=nbpml)
    # Derive timestepping from model spacing
    dt = model.critical_dt
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)
    time = np.linspace(t0, tn, nt)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', ndim=model.dim, f0=0.015, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    # src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, lust below surface)
    rec = Receiver(name='nrec', ntime=nt, npoint=nrec, ndim=model.dim)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    return AnisotropicWaveSolver(model, source=src, receiver=rec,
                                 time_order=time_order,
                                 space_order=space_order, **kwargs)


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        time_order=2, space_order=4, nbpml=10, **kwargs):
    autotune = kwargs.pop('autotune', False)
    solver = tti_setup(dimensions, spacing, tn, time_order, space_order, nbpml, **kwargs)

    if space_order % 2 != 0:
        warning('WARNING: TTI requires a space_order that is a multiple of 4!')

    rec, u, v, summary = solver.forward(autotune=autotune)

    return summary.gflopss, summary.oi, summary.timings, [rec, u, v]


if __name__ == "__main__":
    description = ("Example script to execute a TTI forward operator.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=True, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-to", "--time_order", default=2,
                        type=int, help="Time order of the simulation")
    parser.add_argument("-so", "--space_order", default=8,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    args = parser.parse_args()

    # 3D preset parameters
    if args.dim2:
        dimensions = (150, 150)
        spacing = (10.0, 10.0)
        tn = 500.0
    else:
        dimensions = (100, 100, 100)
        spacing = (10.0, 10.0, 10.0)
        tn = 400.0

    run(dimensions=dimensions, spacing=spacing, nbpml=args.nbpml, tn=tn,
        space_order=args.space_order, time_order=args.time_order,
        autotune=args.autotune, dse='advanced', dle='advanced')

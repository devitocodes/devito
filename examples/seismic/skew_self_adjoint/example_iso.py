import numpy as np
from argparse import ArgumentParser

from devito.logger import info
from devito import configuration, smooth, Function

from examples.seismic import setup_geometry, Model
from examples.seismic.skew_self_adjoint import (setup_w_over_q,
                                                SsaIsoAcousticWaveSolver)


def acousticssa_setup(shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0),
                      tn=500., space_order=4, nbl=10, **kwargs):
    # SSA parameters
    qmin = 0.1
    qmax = 1000.0
    tmax = 500.0
    fpeak = 0.010
    omega = 2.0 * np.pi * fpeak
    vp = 1.5
    b = 1.0

    init_damp = lambda func, nbl: setup_w_over_q(func, omega, qmin, qmax, nbl, sigma=0)
    o = tuple([0]*len(shape))
    spacing = spacing[:len(shape)]
    model = Model(origin=o, shape=shape, vp=vp, b=b, spacing=spacing, nbl=nbl,
                  space_order=space_order, bcs=init_damp, **kwargs)
    # Source and receiver geometries
    geometry = setup_geometry(model, tmax)

    # Create solver object to provide relevant operators
    solver = SsaIsoAcousticWaveSolver(model, geometry,
                                      space_order=space_order, **kwargs)
    return solver


def run(shape=(50, 50, 50), spacing=(10.0, 10.0, 10.0), tn=1000.0,
        space_order=4, nbl=40, full_run=False, autotune=False, **kwargs):

    solver = acousticssa_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                               space_order=space_order, **kwargs)

    info("Applying Forward")
    # Define receiver geometry (spread across x, just below surface)
    rec, u, summary = solver.forward(save=full_run, autotune=autotune)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u.data]

    # Smooth velocity
    initial_vp = Function(name='v0', grid=solver.model.grid, space_order=space_order)
    smooth(initial_vp, solver.model.vp)
    dm = solver.model.vp - initial_vp

    info("Applying Adjoint")
    solver.adjoint(rec, autotune=autotune)
    info("Applying Born")
    solver.born(dm, autotune=autotune)
    info("Applying Gradient")
    solver.gradient(rec, u, autotune=autotune)
    return summary.gflopss, summary.oi, summary.timings, [rec, u.data]


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument("-nd", dest="ndim", default=3, type=int,
                        help="Number of dimensions")
    parser.add_argument("-d", "--shape", default=(51, 51, 51), type=int, nargs="+",
                        help="Number of grid points along each axis")
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("-opt", default="advanced",
                        choices=configuration._accepted['opt'],
                        help="Performance optimization level")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    args = parser.parse_args()

    # 3D preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [15.0])
    tn = 750. if ndim < 3 else 250.

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, autotune=args.autotune,
        space_order=args.space_order, opt=args.opt, full_run=args.full)

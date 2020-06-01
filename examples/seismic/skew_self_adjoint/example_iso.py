import numpy as np

from devito.logger import info
from devito import smooth, Function

from examples.seismic import setup_geometry, Model, seismic_args
from examples.seismic.skew_self_adjoint import (setup_w_over_q,
                                                SsaIsoAcousticWaveSolver)


def acoustic_ssa_setup(shape=(50, 50, 50), spacing=(10.0, 10.0, 10.0),
                       tn=500., space_order=8, nbl=10, **kwargs):
    # SSA parameters
    qmin = 0.1
    qmax = 1000.0
    tmax = 500.0
    fpeak = 0.010
    omega = 2.0 * np.pi * fpeak
    vp = 1.5*np.ones(shape)
    b = 1.0*np.ones(shape)

    init_damp = lambda func, nbl: setup_w_over_q(func, omega, qmin, qmax, nbl, sigma=0)
    o = tuple([0]*len(shape))
    spacing = spacing[:len(shape)]
    model = Model(origin=o, shape=shape, vp=vp, b=b, spacing=spacing, nbl=nbl,
                  space_order=space_order, bcs=init_damp,
                  dtype=kwargs.pop('dtype', np.float32), **kwargs)
    # Source and receiver geometries
    geometry = setup_geometry(model, tmax)

    # Create solver object to provide relevant operators
    solver = SsaIsoAcousticWaveSolver(model, geometry,
                                      space_order=space_order, **kwargs)
    return solver


def run(shape=(50, 50, 50), spacing=(10.0, 10.0, 10.0), tn=1000.0,
        space_order=4, nbl=40, full_run=False, autotune=False, **kwargs):

    solver = acoustic_ssa_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
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
    solver.jacobain(dm, autotune=autotune)
    info("Applying Gradient")
    solver.jacobain_adjoint(rec, u, autotune=autotune)
    return summary.gflopss, summary.oi, summary.timings, [rec, u.data]


if __name__ == "__main__":
    description = ("Example script for a set of SSA isotropic-acoustic operators.")
    args = seismic_args(description)

    # 3D preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [15.0])
    tn = 750. if ndim < 3 else 250.

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, autotune=args.autotune,
        space_order=args.space_order, opt=args.opt, full_run=args.full)

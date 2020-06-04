import numpy as np

from devito.logger import info
from devito import Constant, Function, smooth
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, setup_geometry, seismic_args


def acoustic_setup(shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0),
                   tn=500., kernel='OT2', space_order=4, nbl=10,
                   preset='layers-isotropic', **kwargs):
    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing,
                       **kwargs)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, geometry, kernel=kernel,
                                space_order=space_order, **kwargs)
    return solver


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        space_order=4, kernel='OT2', nbl=40, full_run=False,
        autotune=False, preset='layers-isotropic', checkpointing=False, **kwargs):

    solver = acoustic_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                            space_order=space_order, kernel=kernel,
                            preset=preset, **kwargs)

    info("Applying Forward")
    # Whether or not we save the whole time history. We only need the full wavefield
    # with 'save=True' if we compute the gradient without checkpointing, if we use
    # checkpointing, PyRevolve will take care of the time history
    save = full_run and not checkpointing
    # Define receiver geometry (spread across x, just below surface)
    rec, u, summary = solver.forward(save=save, autotune=autotune)

    if preset == 'constant':
        # With  a new m as Constant
        v0 = Constant(name="v", value=2.0, dtype=np.float32)
        solver.forward(save=save, vp=v0)
        # With a new vp as a scalar value
        solver.forward(save=save, vp=2.0)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u.data]

    # Smooth velocity
    initial_vp = Function(name='v0', grid=solver.model.grid, space_order=space_order)
    smooth(initial_vp, solver.model.vp)
    dm = np.float32(initial_vp.data**(-2) - solver.model.vp.data**(-2))

    info("Applying Adjoint")
    solver.adjoint(rec, autotune=autotune)
    info("Applying Born")
    solver.jacobian(dm, autotune=autotune)
    info("Applying Gradient")
    solver.jacobian_adjoint(rec, u, autotune=autotune, checkpointing=checkpointing)
    return summary.gflopss, summary.oi, summary.timings, [rec, u.data]


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    args = seismic_args(description)
    # 3D preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [15.0])
    tn = 750. if ndim < 3 else 250.

    preset = 'constant-isotropic' if args.constant else 'layers-isotropic'
    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn,
        space_order=args.space_order, preset=preset, kernel=args.kernel,
        autotune=args.autotune, opt=args.opt, full_run=args.full,
        checkpointing=args.checkpointing)

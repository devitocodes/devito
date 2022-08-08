import numpy as np
import pytest

from devito import Function, smooth, norm, info, Constant

from examples.seismic import demo_model, setup_geometry, seismic_args
from examples.seismic.tti import AnisotropicWaveSolver


def tti_setup(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
              kernel='centered', space_order=4, nbl=10, preset='layers-tti',
              **kwargs):

    # Two layer model for true velocity
    model = demo_model(preset, shape=shape, spacing=spacing,
                       space_order=space_order, nbl=nbl, **kwargs)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    return AnisotropicWaveSolver(model, geometry, space_order=space_order,
                                 kernel=kernel, **kwargs)


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        autotune=False, space_order=4, nbl=10, preset='layers-tti',
        kernel='centered', full_run=False, checkpointing=False, **kwargs):

    solver = tti_setup(shape=shape, spacing=spacing, tn=tn, space_order=space_order,
                       nbl=nbl, kernel=kernel, preset=preset, **kwargs)

    info("Applying Forward")
    # Whether or not we save the whole time history. We only need the full wavefield
    # with 'save=True' if we compute the gradient without checkpointing, if we use
    # checkpointing, PyRevolve will take care of the time history
    save = full_run and not checkpointing
    # Define receiver geometry (spread across `x, y`` just below surface)
    rec, u, v, summary = solver.forward(save=save, autotune=autotune)

    if preset in ['constant-tti', 'constant-tti-noazimuth']:
        # With new physical parameters as scalars (slightly higher from original values)
        vp = 2.
        epsilon = .35
        delta = .25
        theta = .75
        phi = None
        if len(shape) > 2 and preset not in ['constant-tti-noazimuth']:
            phi = .4
        solver.forward(save=save, vp=vp, epsilon=epsilon, delta=delta,
                       theta=theta, phi=phi)
        # With new physical parameters as Constants
        d = {'vp': vp, 'epsilon': epsilon, 'delta': delta, 'theta': theta, 'phi': phi}
        d = {k: Constant(name=k, value=v, dtype=np.float32) for k, v in d.items()}
        solver.forward(save=save, **d)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u, v]

    # Smooth velocity
    initial_vp = Function(name='v0', grid=solver.model.grid, space_order=space_order)
    smooth(initial_vp, solver.model.vp)
    dm = np.float32(initial_vp.data**(-2) - solver.model.vp.data**(-2))

    info("Applying Adjoint")
    solver.adjoint(rec, autotune=autotune)
    info("Applying Born")
    solver.jacobian(dm, autotune=autotune)
    info("Applying Gradient")
    solver.jacobian_adjoint(rec, u, v, autotune=autotune, checkpointing=checkpointing)

    return summary.gflopss, summary.oi, summary.timings, [rec, u, v]


@pytest.mark.parametrize('shape', [(51, 51), (16, 16, 16)])
@pytest.mark.parametrize('kernel', ['centered', 'staggered'])
def test_tti_stability(shape, kernel):
    spacing = tuple([20]*len(shape))
    _, _, _, [rec, _, _] = run(shape=shape, spacing=spacing, kernel=kernel,
                               tn=16000.0, nbl=0)
    assert np.isfinite(norm(rec))


if __name__ == "__main__":
    description = ("Example script to execute a TTI forward operator.")
    parser = seismic_args(description)
    parser.add_argument('--noazimuth', dest='azi', default=False, action='store_true',
                        help="Whether or not to use an azimuth angle")
    parser.add_argument("-k", dest="kernel", default='centered',
                        choices=['centered', 'staggered'],
                        help="Choice of finite-difference kernel")
    args = parser.parse_args()

    if args.constant:
        if args.azi:
            preset = 'constant-tti-noazimuth'
        else:
            preset = 'constant-tti'
    else:
        if args.azi:
            preset = 'layers-tti-noazimuth'
        else:
            preset = 'layers-tti'

    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [20.0])
    tn = args.tn if args.tn > 0 else (750. if ndim < 3 else 1250.)

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn,
        space_order=args.space_order, autotune=args.autotune, dtype=args.dtype,
        opt=args.opt, kernel=args.kernel, preset=preset,
        checkpointing=args.checkpointing, full_run=args.full)

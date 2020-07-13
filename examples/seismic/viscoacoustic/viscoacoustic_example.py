import numpy as np
import pytest

from devito.logger import info
from devito import norm
from examples.seismic.viscoacoustic import ViscoacousticWaveSolver
from examples.seismic import demo_model, setup_geometry, seismic_args


def viscoacoustic_setup(shape=(50, 50), spacing=(15.0, 15.0), tn=500., space_order=4,
                        nbl=40, constant=True, kernel='blanch_symes', **kwargs):

    preset = 'constant-viscoacoustic' if constant else 'layers-viscoacoustic'
    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = ViscoacousticWaveSolver(model, geometry, space_order=space_order,
                                     kernel=kernel, **kwargs)
    return solver


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbl=40, autotune=False, constant=False,
        kernel='blanch_symes', **kwargs):

    solver = viscoacoustic_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                                 space_order=space_order, constant=constant,
                                 kernel=kernel, **kwargs)
    info("Applying Forward")

    # Define receiver geometry (spread across x, just below surface)
    rec, p, summary = solver.forward(autotune=autotune)

    return (summary.gflopss, summary.oi, summary.timings, [rec])


def test_viscoacoustic():
    _, _, _, [rec] = run()
    assert np.isclose(norm(rec), 18.7749, atol=1e-3, rtol=0)


@pytest.mark.parametrize('ndim', [1, 2, 3])
@pytest.mark.parametrize('kernel', ['blanch_symes', 'ren', 'deng_mcmechan'])
def test_viscoacoustic_stability(ndim, kernel):
    shape = tuple([11]*ndim)
    spacing = tuple([20]*ndim)
    _, _, _, [rec] = run(shape=shape, spacing=spacing, tn=20000.0, nbl=0, kernel=kernel)
    assert np.isfinite(norm(rec))


if __name__ == "__main__":
    description = ("Example script for a set of viscoacoustic operators.")
    parser = seismic_args(description)
    parser.add_argument("-k", dest="kernel", default='blanch_symes',
                        choices=['blanch_symes', 'ren', 'deng_mcmechan'],
                        help="Choice of finite-difference kernel")
    args = parser.parse_args()
    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [10.0])
    tn = args.tn if args.tn > 0 else (750. if ndim < 3 else 1250.)

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, opt=args.opt,
        space_order=args.space_order, autotune=args.autotune, constant=args.constant,
        kernel=args.kernel)

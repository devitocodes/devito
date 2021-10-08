import numpy as np
import pytest
from devito import norm
from devito.logger import info
from examples.seismic.elastic import ElasticWaveSolver
from examples.seismic import demo_model, setup_geometry, seismic_args


def elastic_setup(shape=(50, 50), spacing=(15.0, 15.0), tn=500., space_order=4,
                  nbl=10, constant=False, **kwargs):

    preset = 'constant-elastic' if constant else 'layers-elastic'
    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = ElasticWaveSolver(model, geometry, space_order=space_order, **kwargs)
    return solver


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbl=40, autotune=False, constant=False, **kwargs):

    solver = elastic_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                           space_order=space_order, constant=constant, **kwargs)
    info("Applying Forward")
    # Define receiver geometry (spread across x, just below surface)
    rec1, rec2, v, tau, summary = solver.forward(autotune=autotune)
    return (summary.gflopss, summary.oi, summary.timings,
            [rec1, rec2, v, tau])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_elastic(dtype):
    _, _, _, [rec1, rec2, v, tau] = run(dtype=dtype)
    assert np.isclose(norm(rec1), 19.25636, atol=1e-3, rtol=0)
    assert np.isclose(norm(rec2), 0.627606, atol=1e-3, rtol=0)


@pytest.mark.parametrize('shape', [(101,), (51, 51), (16, 16, 16)])
def test_elastic_stability(shape):
    spacing = tuple([20]*len(shape))
    _, _, _, [rec1, rec2, v, tau] = run(shape=shape, spacing=spacing, tn=20000.0, nbl=0)
    assert np.isfinite(norm(rec1))


if __name__ == "__main__":
    description = ("Example script for a set of elastic operators.")
    args = seismic_args(description).parse_args()
    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [10.0])
    tn = args.tn if args.tn > 0 else (750. if ndim < 3 else 1250.)

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, opt=args.opt,
        space_order=args.space_order, autotune=args.autotune, constant=args.constant,
        dtype=args.dtype)

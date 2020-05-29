import numpy as np

from devito.logger import info
from examples.seismic.viscoelastic import ViscoelasticWaveSolver
from examples.seismic import demo_model, setup_geometry, seismic_args


def viscoelastic_setup(shape=(50, 50), spacing=(15.0, 15.0), tn=500., space_order=4,
                       nbl=10, constant=True, **kwargs):

    preset = 'constant-viscoelastic' if constant else 'layers-viscoelastic'
    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = ViscoelasticWaveSolver(model, geometry, space_order=space_order, **kwargs)
    return solver


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbl=40, autotune=False, constant=False, **kwargs):

    solver = viscoelastic_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                                space_order=space_order, constant=constant, **kwargs)
    info("Applying Forward")
    # Define receiver geometry (spread across x, just below surface)
    rec1, rec2, v, tau, summary = solver.forward(autotune=autotune)

    return (summary.gflopss, summary.oi, summary.timings,
            [rec1, rec2, v, tau])


def test_viscoelastic():
    _, _, _, [rec1, rec2, v, tau] = run()
    norm = lambda x: np.linalg.norm(x.data.reshape(-1))
    assert np.isclose(norm(rec1), 11.8016, atol=1e-3, rtol=0)
    assert np.isclose(norm(rec2), .276805, atol=1e-3, rtol=0)


if __name__ == "__main__":
    description = ("Example script for a set of viscoelastic operators.")
    args = seismic_args(description)

    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [10.0])
    tn = 750. if ndim < 3 else 1250.

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, opt=args.opt,
        space_order=args.space_order, autotune=args.autotune, constant=args.constant)

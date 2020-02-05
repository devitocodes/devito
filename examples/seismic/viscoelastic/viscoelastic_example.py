import numpy as np
from argparse import ArgumentParser

from devito import configuration
from devito.logger import info
from examples.seismic.viscoelastic import ViscoelasticWaveSolver
from examples.seismic import demo_model, setup_geometry


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
    assert np.isclose(norm(rec1), 13.0748, atol=1e-3, rtol=0)
    assert np.isclose(norm(rec2), 0.43070, atol=1e-3, rtol=0)


if __name__ == "__main__":
    description = ("Example script for a set of viscoelastic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=False, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced", choices=["noop", "advanced"],
                        help="Devito loop engine (DLEE) mode")
    parser.add_argument("--constant", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    args = parser.parse_args()

    # 2D preset parameters
    if args.dim2:
        shape = (150, 150)
        spacing = (10.0, 10.0)
        tn = 750.0
    # 3D preset parameters
    else:
        shape = (150, 150, 150)
        spacing = (10.0, 10.0, 10.0)
        tn = 1250.0

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, dle=args.dle,
        space_order=args.space_order, autotune=args.autotune, constant=args.constant,
        dse=args.dse)

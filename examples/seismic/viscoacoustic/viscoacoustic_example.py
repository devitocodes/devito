import numpy as np
from argparse import ArgumentParser
from devito.logger import info
from devito import configuration
from examples.seismic.viscoacoustic import ViscoacousticWaveSolver
from examples.seismic import demo_model, setup_geometry


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
    norm = lambda x: np.linalg.norm(x.data.reshape(-1))
    assert np.isclose(norm(rec), 24.0908, atol=1e-3, rtol=0)


if __name__ == "__main__":
    description = ("Example script for a set of viscoacoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument("-nd", dest='ndim', default=3, type=int,
                        help="Preset to determine the number of dimensions")
    parser.add_argument("-d", "--shape", default=(150, 150, 150), type=int, nargs="+",
                        help="Determine the grid shape")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("--constant", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    parser.add_argument("-opt", default="advanced",
                        choices=configuration._accepted['opt'],
                        help="Performance optimization level")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    parser.add_argument("-k", dest="kernel", default='blanch_symes',
                        choices=['blanch_symes', 'ren', 'deng_mcmechan'],
                        help="Selects a visco-acoustic equation from the options: \
                        'blanch_symes' - Blanch and Symes (1995) / \
                        Dutta and Schuster (2014) viscoacoustic equation. \
                        'ren' - Ren et al. (2014) viscoacoustic equation. \
                        'deng_mcmechan' - Deng and McMechan (2007) \
                        viscoacoustic equation. Defaults to 'blanch_symes'")
    args = parser.parse_args()

    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [10.0])
    tn = 750. if ndim < 3 else 1250.

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, opt=args.opt,
        space_order=args.space_order, autotune=args.autotune, constant=args.constant,
        kernel=args.kernel)

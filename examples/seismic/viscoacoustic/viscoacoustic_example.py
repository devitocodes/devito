import numpy as np
from argparse import ArgumentParser
from devito.logger import info
from devito import configuration
from examples.seismic.viscoacoustic import ViscoacousticWaveSolver
from examples.seismic import demo_model, setup_geometry

def viscoacoustic_setup(shape=(50, 50), spacing=(15.0, 15.0), tn=500., space_order=4,
                       nbl=40, constant=True, equation=1, **kwargs):
    
    preset = 'constant-viscoacoustic' if constant else 'layers-viscoacoustic'
    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = ViscoacousticWaveSolver(model, geometry, space_order=space_order, 
                                    equation=equation, **kwargs)
    return solver


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbl=40, autotune=False, constant=False, equation=1, **kwargs):

    solver = viscoacoustic_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                                space_order=space_order, constant=constant, 
                                equation=equation, **kwargs)
    info("Applying Forward")

    # Define receiver geometry (spread across x, just below surface)
    rec, p, summary = solver.forward(autotune=autotune)

    return (summary.gflopss, summary.oi, summary.timings, [rec])
    

if __name__ == "__main__":
    description = ("Example script for a set of viscoacoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=False, action='store_true',
                        help="Preset to determine the physical problem setup")
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
    parser.add_argument("--equation", default=1,
            type=int, help="Selects a visco-acoustic equation from the options: \
                            1 - Blanch and Symes (1995) / Dutta and Schuster (2014) \
                            viscoacoustic equation. \
                            2 - Ren et al. (2014) viscoacoustic equation. \
                            3 - Deng and McMechan (2007) viscoacoustic equation. \
                            Defaults to 1.")
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

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, opt=args.opt,
        space_order=args.space_order, autotune=args.autotune, constant=args.constant,
        equation=args.equation)
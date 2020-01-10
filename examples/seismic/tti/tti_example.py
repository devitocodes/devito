import numpy as np
from argparse import ArgumentParser
from devito import configuration
from examples.seismic import demo_model, AcquisitionGeometry
from examples.seismic.tti import AnisotropicWaveSolver


def tti_setup(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
              space_order=4, nbl=10, preset='layers-tti', **kwargs):

    nrec = 101
    # Two layer model for true velocity
    model = demo_model(preset, shape=shape, spacing=spacing, nbl=nbl)
    # Source and receiver geometries
    src_coordinates = np.empty((1, len(spacing)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    if len(shape) > 1:
        src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    if len(shape) > 1:
        rec_coordinates[:, 1] = np.array(model.domain_size)[1] * .5
        rec_coordinates[:, -1] = model.origin[-1] + 2 * spacing[-1]
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=tn, src_type='Ricker', f0=0.010)

    return AnisotropicWaveSolver(model, geometry,
                                 space_order=space_order, **kwargs)


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        autotune=False, time_order=2, space_order=4, nbl=10,
        kernel='centered', **kwargs):

    solver = tti_setup(shape, spacing, tn, space_order, nbl, **kwargs)

    rec, u, v, summary = solver.forward(autotune=autotune, kernel=kernel)

    return summary.gflopss, summary.oi, summary.timings, [rec, u, v]


if __name__ == "__main__":
    description = ("Example script to execute a TTI forward operator.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=False, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument('--noazimuth', dest='azi', default=False, action='store_true',
                        help="Whether or not to use an azimuth angle")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("-k", dest="kernel", default='centered',
                        choices=['centered', 'staggered'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced", choices=["noop", "advanced"],
                        help="Devito loop engine (DLE) mode")
    args = parser.parse_args()

    preset = 'layers-tti-noazimuth' if args.azi else 'layers-tti'
    # 3D preset parameters
    if args.dim2:
        shape = (150, 150)
        spacing = (10.0, 10.0)
        tn = 750.0
    else:
        shape = (50, 50, 50)
        spacing = (10.0, 10.0, 10.0)
        tn = 250.0

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn,
        space_order=args.space_order, autotune=args.autotune, dse=args.dse,
        dle=args.dle, kernel=args.kernel, preset=preset)

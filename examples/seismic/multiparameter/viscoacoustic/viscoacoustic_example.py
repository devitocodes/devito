import numpy as np
import pytest

from devito.logger import info
from devito import norm
from examples.seismic.multiparameter.viscoacoustic import ViscoacousticWaveSolver
from examples.seismic import demo_model, setup_geometry, seismic_args, Receiver


def viscoacoustic_setup(shape=(50, 50), spacing=(15.0, 15.0), tn=500., space_order=4,
                        nbl=40, preset='layers-viscoacoustic', kernel='sls',
                        src_type='ricker', time_order=2, **kwargs):
# Dgauss
    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = ViscoacousticWaveSolver(model, geometry, space_order=space_order,
                                     kernel=kernel, time_order=time_order, **kwargs)
    return solver


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbl=40, autotune=False, preset='layers-viscoacoustic',
        kernel='sls', time_order=2, out_dir=None, **kwargs):

    solver = viscoacoustic_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                                 space_order=space_order, preset=preset,
                                 kernel=kernel, time_order=time_order, **kwargs)
    info("Applying Forward")

    rec, p, v, summary = solver.forward(autotune=autotune)

    return (summary.gflopss, summary.oi, summary.timings, [rec, p])


@pytest.mark.parametrize('kernel, time_order, normrec, atol', [
    ('sls', 2, 1152.5697, 1e-2),
    ('sls', 1, 50.2239, 1e-2),
    ('ren', 2, 1178.0308, 1e-2),
    ('ren', 1, 49.7877, 1e-2),
    ('deng_mcmechan', 2, 1124.141, 1e-2),
    ('deng_mcmechan', 1, 49.3869, 1e-2),
])
def test_viscoacoustic(kernel, time_order, normrec, atol):
    _, _, _, [rec, _] = run(kernel=kernel, time_order=time_order)
    assert np.isclose(norm(rec), normrec, atol=atol, rtol=0)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('kernel', ['sls', 'ren', 'deng_mcmechan'])
@pytest.mark.parametrize('time_order', [1, 2])
def test_viscoacoustic_stability(ndim, kernel, time_order):
    shape = tuple([11]*ndim)
    spacing = tuple([20]*ndim)
    _, _, _, [rec, _] = run(shape=shape, spacing=spacing, tn=20000.0, nbl=0,
                            kernel=kernel, time_order=time_order)
    assert np.isfinite(norm(rec))


def test_adjoint_viscoacoustic():
    tn = 500.  # Final time
    nbl = 10
    shape = (60, 70)
    spacing = (15., 15.)
    space_order = 12
    constant = False
    kernel = 'sls'
    time_order = 1
    src_type = 'Dgauss'
    solver = viscoacoustic_setup(shape=shape, spacing=spacing, tn=tn, src_type=src_type,
                                 space_order=space_order, nbl=nbl, constant=constant,
                                 kernel=kernel, time_order=time_order, dtype=np.float64)
    # Create adjoint receiver symbol
    srca = Receiver(name='srca', grid=solver.model.grid,
                    time_range=solver.geometry.time_axis,
                    coordinates=solver.geometry.src_positions)

    # Run forward and adjoint operators
    rec, _, _ = solver.forward(save=False)
    solver.adjoint(rec=rec, srca=srca)

    # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
    term1 = np.dot(srca.data.reshape(-1), solver.geometry.src.data)
    term2 = np.linalg.norm(rec.data) ** 2

    info('<Ax,y>: %f, <x, A^Ty>: %f, difference: %4.4e, ratio: %f'
         % (term1, term2, (term1 - term2)/term1, term1 / term2))
    assert np.isclose((term1 - term2)/term1, 0., atol=1.e-11)


if __name__ == "__main__":
    description = ("Example script for a set of viscoacoustic operators.")
    parser = seismic_args(description)
    parser.add_argument("-k", dest="kernel", default='sls',
                        choices=['sls', 'ren', 'deng_mcmechan'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-to", "--time_order", default=2,
                        type=int, help="Time order of the equation")
    parser.add_argument("-out_dir", default="./", help="output data directory")
    args = parser.parse_args()
    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [10.0])
    tn = args.tn if args.tn > 0 else (750. if ndim < 3 else 1250.)
    preset = 'constant-viscoacoustic' if args.constant else 'layers-viscoacoustic'

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, opt=args.opt,
        space_order=args.space_order, autotune=args.autotune, preset=preset,
        kernel=args.kernel, time_order=args.time_order, out_dir=args.out_dir)

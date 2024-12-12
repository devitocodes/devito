import argparse
from memory_profiler import memory_usage

import numpy as np
from scipy.ndimage import gaussian_filter

from examples.seismic import (
    Model, AcquisitionGeometry, Receiver, plot_image
)
from examples.seismic.acoustic import AcousticWaveSolver
from devito import configuration, Function, norm, Eq, Operator

configuration['log-level'] = 'WARNING'

# Common Configuration
nshots = 30  # Number of shots
nreceivers = 300  # Number of receivers per shot


# Define the water layer mask
def mask(model, value):
    mask = model > value
    return mask.astype(int)


# Compute residual
def compute_residual(residual, dobs, dsyn):
    if residual.grid.distributor.is_parallel:
        assert np.allclose(dobs.coordinates.data[:], dsyn.coordinates.data)
        assert np.allclose(residual.coordinates.data[:], dsyn.coordinates.data)
        diff_eq = Eq(
            residual, dsyn.subs({dsyn.dimensions[-1]: residual.dimensions[-1]}) -
            dobs.subs({dobs.dimensions[-1]: residual.dimensions[-1]})
        )
        Operator(diff_eq)()
    else:
        residual.data[:] = dsyn.data[:] - dobs.data[:]
    return residual


# FWI Gradient Kernel
def fwi_gradient(mode, model, solver, geometry, source_locations, vp_in, factor=None):
    grad = Function(name="grad", grid=model.grid)
    residual = Receiver(name='residual', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    d_obs = Receiver(name='d_obs', grid=model.grid,
                     time_range=geometry.time_axis,
                     coordinates=geometry.rec_positions)
    d_syn = Receiver(name='d_syn', grid=model.grid,
                     time_range=geometry.time_axis,
                     coordinates=geometry.rec_positions)
    objective = 0.0
    for i in range(nshots):
        geometry.src_positions[0, :] = source_locations[i, :]
        solver.forward(vp=model.vp, rec=d_obs)
        save_value = True if mode == "full" else False
        _, u0, _ = solver.forward(vp=vp_in, save=save_value, rec=d_syn, factor=factor)

        compute_residual(residual, d_obs, d_syn)
        objective += 0.5 * norm(residual)**2

        solver.gradient(rec=residual, u=u0, vp=vp_in, grad=grad, factor=factor)

    return objective, grad


def main(mode, factor=None):
    shape = (601, 221)
    spacing = (15.0, 15.0)
    origin = (0.0, 0.0)
    vel_path = '../../../devito/data/Marm.bin'
    model_true = np.fromfile(vel_path, np.float32).reshape(221, 601)
    msk = mask(model_true, 1.5)
    model_init = gaussian_filter(model_true, sigma=[10, 15]) * msk
    model_init[model_init == 0] = 1.5

    model = Model(vp=model_true.T, origin=origin, shape=shape, spacing=spacing,
                  space_order=2, nbl=20, bcs="damp")
    model0 = Model(vp=model_init.T, origin=origin, shape=shape, spacing=spacing,
                   space_order=2, nbl=20, bcs="damp", grid=model.grid)

    t0, tn, f0 = 0.0, 4000.0, 0.005
    src_coordinates = np.array([[model.domain_size[0] * 0.5, 20.0]])
    rec_coordinates = np.column_stack((
        np.linspace(0, model.domain_size[0], nreceivers),
        np.full(nreceivers, 20.0)
    ))
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0, tn, f0=f0, src_type='Ricker')

    solver = AcousticWaveSolver(model, geometry, space_order=4)

    source_locations = np.column_stack((
        np.linspace(0.0, model.domain_size[0], nshots),
        np.zeros(nshots)
    ))

    ff, grad = fwi_gradient(mode, model, solver, geometry,
                            source_locations, model0.vp, factor)
    mem_usage = memory_usage()[0]
    print(f"Memory usage at the end of gradient ({mode} mode): {mem_usage:.2f} MiB")
    grad_max = np.abs(grad.data[:]).max()
    plot_image(-grad.data / grad_max, vmin=-1, vmax=1, cmap="seismic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FWI Gradient Computation Modes")
    parser.add_argument(
        "--mode", choices=["full", "snapshot"], required=True,
        help="Choose the mode: 'full' for full time axis or 'snapshot' for snapshots"
    )
    parser.add_argument(
        "--factor", type=int, default=None,
        help="Snapshot saving factor (only relevant for snapshot mode)"
    )
    args = parser.parse_args()
    main(args.mode, args.factor)

import numpy as np
from argparse import ArgumentParser

from devito.logger import info
from devito import Constant
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, RickerSource, Receiver


# Velocity models
def smooth10(vel, shape):
    if np.isscalar(vel):
        return .9 * vel * np.ones(shape, dtype=np.float32)
    out = np.ones(shape, dtype=np.float32)
    nz = shape[-1]

    for a in range(5, nz-6):
        if len(shape) == 2:
            out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
        else:
            out[:, :, a] = np.sum(vel[:, :, a - 5:a + 5], axis=2) / 10

    return out


def acoustic_setup(shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0),
                   tn=500., time_order=2, space_order=4, nbpml=10,
                   constant=False, **kwargs):
    nrec = shape[0]
    preset = 'constant-isotropic' if constant else 'layers-isotropic'
    model = demo_model(preset, shape=shape, spacing=spacing, nbpml=nbpml)

    # Derive timestepping from model spacing
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time = np.linspace(t0, tn, nt)  # Discretized time axis

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 5 * spacing[-1]

    # Define receiver geometry (spread across x, just below surface)
    rec = Receiver(name='nrec', grid=model.grid, ntime=nt, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                time_order=time_order,
                                space_order=space_order, **kwargs)
    return solver


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        time_order=2, space_order=4, nbpml=40, full_run=False,
        autotune=False, constant=False, **kwargs):

    solver = acoustic_setup(shape=shape, spacing=spacing, nbpml=nbpml, tn=tn,
                            space_order=space_order, time_order=time_order,
                            constant=constant, **kwargs)

    initial_vp = smooth10(solver.model.m.data, solver.model.shape_domain)
    dm = np.float32(initial_vp**2 - solver.model.m.data)
    info("Applying Forward")
    rec, u, summary = solver.forward(save=full_run, autotune=autotune)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(rec.data[:, :], vmin=-10, vmax=10, cmap="seismic", aspect=.2)
    plt.figure()
    plt.imshow(np.transpose(u.data[-1, 90, :, :]), vmin=-1, vmax=1, cmap="seismic", aspect=1)
    plt.show()

    if constant:
        # With  a new m as Constant
        m0 = Constant(name="m", value=.25, dtype=np.float32)
        solver.forward(save=full_run, m=m0)
        # With a new m as a scalar value
        solver.forward(save=full_run, m=.25)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u.data]

    info("Applying Adjoint")
    solver.adjoint(rec, autotune=autotune)
    info("Applying Born")
    solver.born(dm, autotune=autotune)
    info("Applying Gradient")
    solver.gradient(rec, u, autotune=autotune)


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=False, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-to", "--time_order", default=2,
                        type=int, help="Time order of the simulation")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DLE) mode")
    parser.add_argument("--constant", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    args = parser.parse_args()

    # 3D preset parameters
    if args.dim2:
        shape = (150, 150)
        spacing = (15.0, 15.0)
        tn = 500.0
    else:
        shape = (100, 100, 100)
        spacing = (15.0, 15.0, 15.0)
        tn = 500.0

    run(shape=shape, spacing=spacing, nbpml=args.nbpml, tn=tn,
        space_order=args.space_order, time_order=args.time_order, constant=args.constant,
        autotune=args.autotune, dse=args.dse, dle=args.dle, full_run=args.full)

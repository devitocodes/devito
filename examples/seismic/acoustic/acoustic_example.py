import numpy as np
from argparse import ArgumentParser

from devito.logger import info
from devito import Constant
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, TimeAxis, RickerSource, Receiver


# Velocity models
def smooth10(vel, shape):
    """
    Smooth the input n-dimensional array 'vel' along its last dimension (depth)
    with a 10 points moving averaging kernel.
    """
    # Return a scaled version of the input if the input is a scalar
    if np.isscalar(vel):
        return .9 * vel * np.ones(shape, dtype=np.float32)
    # Initialize output
    out = np.copy(vel)
    # Size of the smoothed dimension
    nz = shape[-1]
    # Indexing is done via slices for YASK compatibility
    # Fist get the full span
    full_dims = [slice(0, d) for d in shape[:-1]]
    for a in range(5, nz-6):
        # Get the a-5 yto ai+5 indices along the last dimension at index a
        slicessum = full_dims + [slice(a - 5, a + 5)]
        # Average input
        out[..., a] = np.sum(vel[slicessum], axis=len(shape)-1) / 10

    return out


def acoustic_setup(shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0),
                   tn=500., kernel='OT2', space_order=4, nbpml=10,
                   constant=False, **kwargs):
    nrec = shape[0]
    preset = 'constant-isotropic' if constant else 'layers-isotropic'
    model = demo_model(preset, space_order=space_order, shape=shape, nbpml=nbpml,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing)

    # Derive timestepping from model spacing
    dt = model.critical_dt * (1.73 if kernel == 'OT4' else 1.0)
    t0 = 0.0
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    if len(shape) > 1:
        src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]
    # Define receiver geometry (spread across x, just below surface)
    rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    if len(shape) > 1:
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec, kernel=kernel,
                                space_order=space_order, **kwargs)
    return solver


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        space_order=4, kernel='OT2', nbpml=40, full_run=False,
        autotune=False, constant=False, checkpointing=False, **kwargs):

    solver = acoustic_setup(shape=shape, spacing=spacing, nbpml=nbpml, tn=tn,
                            space_order=space_order, kernel=kernel,
                            constant=constant, **kwargs)

    initial_vp = smooth10(solver.model.m.data, solver.model.shape_domain)
    dm = np.float32(initial_vp**2 - solver.model.m.data)
    info("Applying Forward")
    # Whether or not we save the whole time history. We only need the full wavefield
    # with 'save=True' if we compute the gradient without checkpointing, if we use
    # checkpointing, PyRevolve will take care of the time history
    save = full_run and not checkpointing
    # Define receiver geometry (spread across x, just below surface)
    rec, u, summary = solver.forward(save=save, autotune=autotune)

    if constant:
        # With  a new m as Constant
        m0 = Constant(name="m", value=.25, dtype=np.float32)
        solver.forward(save=save, m=m0)
        # With a new m as a scalar value
        solver.forward(save=save, m=.25)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u.data]

    info("Applying Adjoint")
    solver.adjoint(rec, autotune=autotune)
    info("Applying Born")
    solver.born(dm, autotune=autotune)
    info("Applying Gradient")
    solver.gradient(rec, u, autotune=autotune, checkpointing=checkpointing)


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('-nd', dest='ndim', default=3, type=int,
                        help="Preset to determine the number of dimensions")
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", "-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DSE) mode")
    parser.add_argument("--constant", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    parser.add_argument("--checkpointing", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    args = parser.parse_args()

    # 3D preset parameters
    shape = tuple(args.ndim * [51])
    spacing = tuple(args.ndim * [15.0])
    tn = 750. if args.ndim < 3 else 250.

    run(shape=shape, spacing=spacing, nbpml=args.nbpml, tn=tn,
        space_order=args.space_order, constant=args.constant, kernel=args.kernel,
        autotune=args.autotune, dse=args.dse, dle=args.dle, full_run=args.full,
        checkpointing=args.checkpointing)

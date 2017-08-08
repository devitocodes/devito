import numpy as np

from devito.logger import info
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Model, PointSource, Receiver
from devito import ConstantData


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))

    return (1-2.*r**2)*np.exp(-r**2)


def setup(dimensions=(50, 50, 50), spacing=(15.0, 15.0, 15.0), tn=500.,
          time_order=2, space_order=4, nbpml=10, vp=1.5, **kwargs):

    ndim = len(dimensions)
    origin = tuple([0.]*ndim)
    spacing = spacing[:ndim]
    # Velocity model, two layers
    true_vp = vp
    # Source location
    location = np.zeros((1, ndim), dtype=np.float32)
    location[0, :-1] = [origin[i] + dimensions[i] * spacing[i] * .5
                        for i in range(ndim-1)]
    location[0, -1] = origin[-1] + 2 * spacing[-1]
    # Receivers locations
    receiver_coords = np.zeros((dimensions[0], ndim), dtype=np.float32)
    receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                        (dimensions[0]-1) * spacing[0],
                                        num=dimensions[0])
    receiver_coords[:, 1:] = location[0, 1:]

    # Define seismic data
    model = Model(origin, spacing, dimensions, true_vp, nbpml=int(nbpml))

    f0 = .010
    dt = model.critical_dt
    if time_order == 4:
        dt *= 1.73
    t0 = 0.0
    nt = int(1+(tn-t0)/dt)

    # Set up the source as Ricker wavelet for f0
    def source(t, f0):
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)

    # Source geometry
    time_series = np.zeros((nt, 1), dtype=np.float32)
    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

    # Define source and receivers and create acoustic wave solver
    src = PointSource(name='src', data=time_series, coordinates=location)
    rec = Receiver(name='rec', ntime=nt, coordinates=receiver_coords)
    return AcousticWaveSolver(model, source=src, receiver=rec,
                              time_order=time_order, space_order=space_order, **kwargs)


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        time_order=2, space_order=4, nbpml=40, full_run=False, **kwargs):

    solver = setup(dimensions=dimensions, spacing=spacing, nbpml=nbpml, tn=tn,
                   space_order=space_order, time_order=time_order, **kwargs)

    initial_vp = 1.8
    dm = (initial_vp**2 - solver.model.m.data) * np.ones(solver.model.shape_domain,
                                                         dtype=np.float32)
    info("Applying Forward")
    # Default model.m
    rec, u, summary = solver.forward(save=full_run)
    # With  a new m as ConstantData
    m0 = ConstantData(name="m", value=.25, dtype=np.float32)
    solver.forward(save=full_run, m=m0)
    # With a new m as a scalar value
    solver.forward(save=full_run, m=.25)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u.data]

    info("Applying Adjoint")
    solver.adjoint(rec, **kwargs)
    info("Applying Born")
    solver.born(dm, **kwargs)
    info("Applying Gradient")
    solver.gradient(rec, u, **kwargs)


if __name__ == "__main__":
    run(full_run=True, autotune=False, space_order=6, time_order=2,
        dimensions=(50, 50), tn=500., nbpml=0)

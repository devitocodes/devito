import numpy as np

from devito.logger import warning
from examples.seismic import Model, GaborSource, Receiver
from examples.seismic.tti import AnisotropicWaveSolver


def setup(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0, time_order=2,
          space_order=4, nbpml=10, dse='advanced', dle='advanced'):

    ndim = len(dimensions)
    nrec = 101
    origin = (0., 0., 0.)

    # Two layer model for true velocity
    model = demo_model('layers', ratio=3,
                       shape=dimensions, spacing=spacing
                       origin=origin, nbpml=nbpml,
                       epsilon=.4*np.ones(dimensions),
                       delta=-.1*np.ones(dimensions),
                       theta=-np.pi/7*np.ones(dimensions),
                       phi=np.pi/5*np.ones(dimensions))

    # Derive timestepping from model spacing
    dt = model.critical_dt
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)
    time = np.linspace(t0, tn, nt)

    # Define source geometry (center of domain, just below surface)
    src = GaborSource(name='src', ndim=ndim, f0=0.01, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, lust below surface)
    rec = Receiver(name='nrec', ntime=nt, npoint=nrec, ndim=ndim)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    return AnisotropicWaveSolver(model, source=src, time_order=time_order,
                                 space_order=space_order, receiver=rec, dse=dse, dle=dle)


def run(dimensions=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        time_order=2, space_order=4, nbpml=10, **kwargs):
    autotune = kwargs.pop('autotune', False)
    solver = setup(dimensions, spacing, tn, time_order, space_order, nbpml, **kwargs)

    if space_order % 4 != 0:
        warning('WARNING: TTI requires a space_order that is a multiple of 4!')

    rec, u, v, summary = solver.forward(autotune=autotune)

    return summary.gflopss, summary.oi, summary.timings, [rec, u, v]


if __name__ == "__main__":
    run()

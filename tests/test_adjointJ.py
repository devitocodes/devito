import numpy as np
import pytest
from numpy import linalg
from conftest import skipif_yask

from devito.logger import info
from examples.seismic import demo_model, TimeAxis, RickerSource, Receiver
from examples.seismic.acoustic import AcousticWaveSolver


@skipif_yask
@pytest.mark.parametrize('space_order', [4, 8, 12])
@pytest.mark.parametrize('shape', [(60, 70), (40, 50, 30)])
def test_acousticJ(shape, space_order):
    t0 = 0.0  # Start time
    tn = 500.  # Final time
    nrec = shape[0]  # Number of receivers
    nbpml = 10 + space_order / 2
    spacing = [15. for _ in shape]

    # Create two-layer "true" model from preset with a fault 1/3 way down
    model = demo_model('layers-isotropic', ratio=3, vp_top=1.5, vp_bottom=2.5,
                       spacing=spacing, space_order=space_order, shape=shape,
                       nbpml=nbpml, dtype=np.float64)

    # Derive timestepping from model spacing
    dt = model.critical_dt
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = 30.

    # Define receiver geometry (same as source, but spread across x)
    rec = Receiver(name='nrec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                kernel='OT2', space_order=space_order)

    # Create initial model (m0) with a constant velocity throughout
    model0 = demo_model('layers-isotropic', ratio=3, vp_top=1.5, vp_bottom=1.5,
                        spacing=spacing, space_order=space_order, shape=shape,
                        nbpml=nbpml, dtype=np.float64)

    # Compute the full wavefield u0
    _, u0, _ = solver.forward(save=True, m=model0.m)

    # Compute initial born perturbation from m - m0
    dm = model.m.data - model0.m.data

    du, _, _, _ = solver.born(dm, m=model0.m)

    # Compute gradientfrom initial perturbation
    im, _ = solver.gradient(du, u0, m=model0.m)

    # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
    term1 = np.dot(im.data.reshape(-1), dm.reshape(-1))
    term2 = linalg.norm(du.data)**2
    info('<Ax,y>: %f, <x, A^Ty>: %f, difference: %12.12f, ratio: %f'
         % (term1, term2, term1 - term2, term1 / term2))
    assert np.isclose(term1 / term2, 1.0, atol=0.001)


if __name__ == "__main__":
    test_acousticJ(shape=(60, 70), space_order=4)

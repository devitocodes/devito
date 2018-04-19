import numpy as np
import pytest
from conftest import skipif_yask
from numpy import linalg

from devito import TimeFunction
from devito.logger import log
from examples.seismic import TimeAxis, PointSource, RickerSource, Receiver, demo_model
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.tti import AnisotropicWaveSolver


@skipif_yask
@pytest.mark.parametrize('shape', [(120, 140), (120, 140, 150)])
@pytest.mark.parametrize('space_order', [4, 8])
def test_tti(shape, space_order):
    nbpml = 10
    ndim = len(shape)
    origin = [0. for _ in shape]
    spacing = [10. for _ in shape]

    # Source location
    location = np.zeros((1, ndim), dtype=np.float32)
    location[0, :-1] = [origin[i] + shape[i] * spacing[i] * .5
                        for i in range(ndim-1)]
    location[0, -1] = origin[-1] + 2 * spacing[-1]

    # Receivers locations
    receiver_coords = np.zeros((shape[0], ndim), dtype=np.float32)
    receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                        (shape[0]-1) * spacing[0],
                                        num=shape[0])
    receiver_coords[:, 1:] = location[0, 1:]

    # Two layer model for true velocity
    model = demo_model('layers-isotropic', ratio=3, shape=shape,
                       spacing=spacing, nbpml=nbpml, space_order=space_order,
                       epsilon=np.zeros(shape), delta=np.zeros(shape),
                       theta=np.zeros(shape), phi=np.zeros(shape))

    # Define seismic data and parameters
    f0 = .010
    dt = model.critical_dt
    t0 = 0.0
    tn = 350.0
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    nt = time_range.num

    last = (nt - 2) % 3
    indlast = [(last + 1) % 3, last % 3, (last-1) % 3]

    # Adjoint test
    source = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range)
    receiver = Receiver(name='rec', grid=model.grid, time_range=time_range,
                        coordinates=receiver_coords)
    acoustic = AcousticWaveSolver(model, source=source, receiver=receiver,
                                  time_order=2, space_order=space_order)
    rec, u1, _ = acoustic.forward(save=False)

    tn = 100.0
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    nt = time_range.num

    # Source geometry
    time_series = np.zeros((nt, 1))

    source = PointSource(name='src', grid=model.grid, time_range=time_range,
                         data=time_series, coordinates=location)
    receiver = Receiver(name='rec', grid=model.grid, time_range=time_range,
                        coordinates=receiver_coords)
    acoustic = AcousticWaveSolver(model, source=source, receiver=receiver,
                                  time_order=2, space_order=space_order)

    solver_tti = AnisotropicWaveSolver(model, source=source, receiver=receiver,
                                       time_order=2, space_order=space_order)

    # Create new wavefield object restart forward computation
    u = TimeFunction(name='u', grid=model.grid,
                     time_order=2, space_order=space_order, dtype=model.dtype)
    u.data[0:3, :] = u1.data[indlast, :]
    rec, _, _ = acoustic.forward(save=False, u=u)

    utti = TimeFunction(name='u', grid=model.grid,
                        time_order=2, space_order=space_order, dtype=model.dtype)
    vtti = TimeFunction(name='v', grid=model.grid,
                        time_order=2, space_order=space_order, dtype=model.dtype)
    utti.data[0:3, :] = u1.data[indlast, :]
    vtti.data[0:3, :] = u1.data[indlast, :]
    rec_tti, u_tti, v_tti, _ = solver_tti.forward(u=utti, v=vtti)

    res = linalg.norm(u.data.reshape(-1) -
                      .5 * u_tti.data.reshape(-1) - .5 * v_tti.data.reshape(-1))
    res /= linalg.norm(u.data.reshape(-1))

    log("Difference between acoustic and TTI with all coefficients to 0 %f" % res)
    assert np.isclose(res, 0.0, atol=1e-4)


if __name__ == "__main__":
    test_tti((120, 140), 4)

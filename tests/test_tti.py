import numpy as np
import pytest
from numpy import linalg

from devito.logger import log
from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot
from examples.tti.TTI_codegen import TTI_cg


@pytest.mark.parametrize('dimensions', [(120, 140), (120, 140, 150)])
@pytest.mark.parametrize('space_order', [4, 8])
def test_tti(dimensions, space_order):
    nbpml = 10

    if len(dimensions) == 2:
        # Dimensions in 2D are (x, z)
        origin = (0., 0.)
        spacing = (10., 10.)

        # Source location
        location = np.zeros((1, 2))
        location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
        location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5

        # Receiver coordinates
        receiver_coords = np.zeros((101, 2))
        receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                            dimensions[0] * spacing[0], num=101)
        receiver_coords[:, 1] = origin[1] + 50 * spacing[1]

    elif len(dimensions) == 3:
        # Dimensions in 3D are (x, y, z)
        origin = (0., 0., 0.)
        spacing = (10., 10., 10.)

        # Source location
        location = np.zeros((1, 3))
        location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
        location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
        location[0, 2] = origin[2] + dimensions[2] * spacing[2] * 0.5

        # Receiver coordinates
        receiver_coords = np.zeros((101, 3))
        receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                            dimensions[0] * spacing[0], num=101)
        receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
        receiver_coords[:, 2] = origin[2] + 50 * spacing[2]

    # True velocity
    true_vp = np.ones(dimensions) + .5

    model = IGrid(origin, spacing,
                  true_vp,
                  rho=0.0 * true_vp,
                  epsilon=0.0 * true_vp,
                  delta=0.0 * true_vp,
                  theta=0.0 * true_vp,
                  phi=0.0 * true_vp)
    # Define seismic data.
    data = IShot()
    src = IShot()

    f0 = .010
    dt = model.get_critical_dt()
    t0 = 0.0
    tn = 350.0
    nt = int(1+(tn-t0)/dt)
    last = (nt - 1) % 3
    indlast = [(last + 1) % 3, (last+2) % 3, last % 3]

    # Set up the source as Ricker wavelet for f0
    def source(t, f0):
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)

    # Source geometry
    time_series = np.zeros((nt, 1))
    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)
    src.set_receiver_pos(location)
    src.set_shape(nt, 1)
    src.set_traces(time_series)

    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, 101)

    # Adjoint test
    wave_acou = Acoustic_cg(model, data, src, t_order=2, s_order=space_order,
                            nbpml=nbpml)
    rec, u1, _, _, _ = wave_acou.Forward(save=False)

    tn = 50.0
    nt = int(1 + (tn - t0) / dt)
    # Source geometry
    time_series = np.zeros((nt, 1))
    time_series[:, 0] = 0*source(np.linspace(t0, tn, nt), f0)
    src.set_shape(nt, 1)
    src.set_traces(time_series)
    data.set_shape(nt, 101)

    wave_acou = Acoustic_cg(model, data, src, t_order=2, s_order=space_order,
                            nbpml=nbpml)

    wave_tti = TTI_cg(model, data, src, t_order=2, s_order=space_order,
                      nbpml=nbpml)

    rec, u, _, _, _ = wave_acou.Forward(save=False, u_ini=u1.data[indlast, :])
    rec_tti, u_tti, v_tti, _, _, _ = wave_tti.Forward(save=False,
                                                      u_ini=u1.data[indlast, :],
                                                      legacy=True)

    res = linalg.norm(u.data.reshape(-1) -
                      .5 * u_tti.reshape(-1) - .5 * v_tti.reshape(-1))
    res /= linalg.norm(u.data.reshape(-1))
    log("Difference between acoustic and TTI with all coefficients to 0 %f" % res)
    assert np.isclose(res, 0.0, atol=1e-1)


if __name__ == "__main__":
    test_tti((120, 140, 130), 4)

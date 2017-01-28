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
    # dimensions are (x,z) and (x, y, z)
    origin = tuple([0.0]*len(dimensions))
    spacing = tuple([15.0]*len(dimensions))

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
    tn = 700.0
    nt = int(1+(tn-t0)/dt)

    # Set up the source as Ricker wavelet for f0
    def source(t, f0):
        agauss = 0.5 * f0
        tcut = 2 / agauss
        s = (t - tcut) * agauss
        return np.exp(-2 * s**2) * np.cos(2 * np.pi * s)

    # Source geometry
    time_series = np.zeros((nt, 1))
    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

    location = np.zeros((1, len(dimensions)))
    location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
    location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    if len(dimensions) == 3:
        location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
        location[0, 2] = origin[2] + dimensions[2] * spacing[2] * 0.5
    src.set_receiver_pos(location)
    src.set_shape(nt, 1)
    src.set_traces(time_series)

    receiver_coords = np.zeros((101, len(dimensions)))
    receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                        dimensions[0] * spacing[0], num=101)
    receiver_coords[:, 1] = origin[1] + 50 * spacing[1]
    if len(dimensions) == 3:
        receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
        receiver_coords[:, 2] = origin[2] + 50 * spacing[2]
    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, 101)

    # Adjoint test
    wave_acou = Acoustic_cg(model, data, src, t_order=2, s_order=space_order,
                            nbpml=10)
    rec, u1, _, _, _ = wave_acou.Forward(save=False)

    tn = 50.0
    nt = int(1 + (tn - t0) / dt)
    # Source geometry
    time_series = np.zeros((nt, 1))
    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)
    src.set_shape(nt, 1)
    src.set_traces(time_series)
    data.set_shape(nt, 101)

    wave_acou = Acoustic_cg(model, data, src, t_order=2, s_order=space_order,
                            nbpml=10)

    wave_tti = TTI_cg(model, data, src, t_order=2, s_order=space_order,
                      nbpml=10)

    rec, u, _, _, _ = wave_acou.Forward(save=False, u_ini=u1.data)
    rec_tti, u_tti, v_tti, _, _, _ = wave_tti.Forward(save=False, u_ini=u1.data)

    res = linalg.norm(u.data.reshape(-1) -
                      .5 * u_tti.reshape(-1) - .5 * v_tti.reshape(-1))
    res /= linalg.norm(u.data.reshape(-1))
    log("Difference between acoustic and TTI with all coefficients to 0 %f" % res)
    assert np.isclose(res, 0.0, atol=1e-1)


if __name__ == "__main__":
    test_tti((120, 140), 4)

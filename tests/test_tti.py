import numpy as np
import pytest
from numpy import linalg

from devito import TimeFunction
from devito.logger import log
from examples.seismic.model import SeismicModel
from examples.seismic.acoustic import acoustic_setup
from examples.seismic.tti import tti_setup


@pytest.mark.parametrize('shape, so, rot', [
    ((60, 70), 4, True), ((60, 70), 8, False),
    ((60, 70, 75), 4, True), ((60, 70, 75), 8, False)])
def test_tti(shape, so, rot):
    """
    This first test compare the solution of the acoustic wave-equation and the
    TTI wave-equation with Thomsen parameters to 0. The two solutions should
    be the same with and without rotation angles (Laplacian is rotational invariant).
    """
    to = 2
    origin = [0. for _ in shape]
    spacing = [20. for _ in shape]
    vp = 1.5 * np.ones(shape)
    rot_val = .01 if rot else 0.

    # Create acoustic solver from preset
    acoustic = acoustic_setup(origin=origin, shape=shape, spacing=spacing,
                              vp=vp, nbl=0, tn=350., space_order=so,
                              preset='constant-isotropic')

    # Create tti solver from preset
    solver_tti = tti_setup(origin=origin, shape=shape, spacing=spacing,
                           vp=vp, nbl=0, tn=350., space_order=so,
                           preset='constant-tti')

    dt = solver_tti.model.critical_dt
    geometry = solver_tti.geometry
    acoustic.geometry.resample(dt)

    rec, u1, _ = acoustic.forward(save=False, dt=dt)

    # zero src
    src = geometry.src
    src.data.fill(0.)
    # last time index
    nt = geometry.nt
    last = (nt - 2) % 3
    indlast = [(last + 1) % 3, last % 3, (last-1) % 3]

    # Create new wavefield object restart forward computation
    u = TimeFunction(name='u', grid=acoustic.model.grid, time_order=2, space_order=so)
    u.data[0:3, :] = u1.data[indlast, :]
    acoustic.forward(save=False, u=u, time_M=10, src=src, dt=dt)

    utti = TimeFunction(name='u', grid=solver_tti.model.grid, time_order=to,
                        space_order=so)
    vtti = TimeFunction(name='v', grid=solver_tti.model.grid, time_order=to,
                        space_order=so)

    utti.data[0:to+1, :] = u1.data[indlast[:to+1], :]
    vtti.data[0:to+1, :] = u1.data[indlast[:to+1], :]

    model = SeismicModel(space_order=so, vp=vp, origin=origin, shape=shape,
                         spacing=spacing, nbl=0, epsilon=0.,
                         delta=0., theta=rot_val, phi=rot_val, bcs="damp")

    solver_tti.forward(u=utti, v=vtti, model=model, time_M=10, src=src, dt=dt)

    normal_u = u.data[:]
    normal_utti = .5 * utti.data[:]
    normal_vtti = .5 * vtti.data[:]

    res = linalg.norm((normal_u - normal_utti - normal_vtti).reshape(-1))**2
    res /= np.linalg.norm(normal_u.reshape(-1))**2
    log("Difference between acoustic and TTI with all coefficients to 0 %2.4e" % res)
    assert np.isclose(res, 0.0, atol=1e-4)

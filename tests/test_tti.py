import numpy as np
import pytest
from numpy import linalg

from devito import TimeFunction
from devito.logger import log
from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.tti import AnisotropicWaveSolver


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
    nrec = shape[0]
    rot_val = .01*np.ones(shape) if rot else np.zeros(shape)

    # Constant model for true velocity
    model = Model(origin=origin, shape=shape, vp=vp,
                  spacing=spacing, nbl=0, space_order=so,
                  epsilon=np.zeros(shape), delta=np.zeros(shape),
                  theta=rot_val, phi=rot_val)

    # Source and receiver geometries
    src_coordinates = np.empty((1, len(spacing)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec_coordinates[:, 1] = np.array(model.domain_size)[1] * .5
    rec_coordinates[:, -1] = model.origin[-1] + 2 * spacing[-1]

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=350., src_type='Ricker', f0=0.005)

    acoustic = AcousticWaveSolver(model, geometry, space_order=so)
    rec, u1, _ = acoustic.forward(save=False)

    # Solvers
    solver_tti = AnisotropicWaveSolver(model, geometry, space_order=so)

    # zero src
    src = geometry.src
    src.data.fill(0.)
    # last time index
    nt = geometry.nt
    last = (nt - 2) % 3
    indlast = [(last + 1) % 3, last % 3, (last-1) % 3]

    # Create new wavefield object restart forward computation
    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=so)
    u.data[0:3, :] = u1.data[indlast, :]
    acoustic.forward(save=False, u=u, time_M=10, src=src)

    utti = TimeFunction(name='u', grid=model.grid, time_order=to, space_order=so)
    vtti = TimeFunction(name='v', grid=model.grid, time_order=to, space_order=so)

    utti.data[0:to+1, :] = u1.data[indlast[:to+1], :]
    vtti.data[0:to+1, :] = u1.data[indlast[:to+1], :]

    solver_tti.forward(u=utti, v=vtti, time_M=10, src=src)

    normal_u = u.data[:]
    normal_utti = .5 * utti.data[:]
    normal_vtti = .5 * vtti.data[:]

    res = linalg.norm((normal_u - normal_utti - normal_vtti).reshape(-1))**2
    res /= np.linalg.norm(normal_u.reshape(-1))**2
    log("Difference between acoustic and TTI with all coefficients to 0 %2.4e" % res)
    assert np.isclose(res, 0.0, atol=1e-4)

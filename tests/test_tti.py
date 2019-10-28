import numpy as np
import pytest
from numpy import linalg

from conftest import skipif
from devito import TimeFunction
from devito.logger import log
from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.tti import AnisotropicWaveSolver

pytestmark = skipif(['yask', 'ops'])


@pytest.mark.parametrize('shape', [(120, 140), (120, 140, 150)])
@pytest.mark.parametrize('space_order', [4, 8])
@pytest.mark.parametrize('kernel', ['centered'])
def test_tti(shape, space_order, kernel):
    """
    This first test compare the solution of the acoustic wave-equation and the
    TTI wave-eqatuon with all anisotropy parametrs to 0. The two solutions should
    be the same.
    """
    if kernel == 'shifted':
        space_order *= 2
    to = 2
    so = space_order
    nbl = 10
    origin = [0. for _ in shape]
    spacing = [10. for _ in shape]
    vp = 1.5 * np.ones(shape)
    nrec = shape[0]

    # Constant model for true velocity
    model = Model(origin=origin, shape=shape, vp=vp,
                  spacing=spacing, nbl=nbl, space_order=space_order,
                  epsilon=np.zeros(shape), delta=np.zeros(shape),
                  theta=np.zeros(shape), phi=np.zeros(shape))

    # Source and receiver geometries
    src_coordinates = np.empty((1, len(spacing)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec_coordinates[:, 1] = np.array(model.domain_size)[1] * .5
    rec_coordinates[:, -1] = model.origin[-1] + 2 * spacing[-1]

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=350., src_type='Ricker', f0=0.010)

    acoustic = AcousticWaveSolver(model, geometry, time_order=2, space_order=so)
    rec, u1, _ = acoustic.forward(save=False)

    # Solvers
    solver_tti = AnisotropicWaveSolver(model, geometry, time_order=2,
                                       space_order=space_order)

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

    solver_tti.forward(u=utti, v=vtti, kernel=kernel, time_M=10, src=src)

    normal_u = u.data[:]
    normal_utti = .5 * utti.data[:]
    normal_vtti = .5 * vtti.data[:]

    res = linalg.norm((normal_u - normal_utti - normal_vtti).reshape(-1))**2
    res /= np.linalg.norm(normal_u.reshape(-1))**2

    log("Difference between acoustic and TTI with all coefficients to 0 %2.4e" % res)
    assert np.isclose(res, 0.0, atol=1e-4)

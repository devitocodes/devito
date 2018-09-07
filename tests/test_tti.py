import numpy as np
import pytest
from conftest import skipif_yask
from numpy import linalg

from devito import TimeFunction, configuration
from devito.logger import log
from examples.seismic import TimeAxis, RickerSource, Receiver, Model, demo_model
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.tti import AnisotropicWaveSolver


@skipif_yask
@pytest.mark.parametrize('shape', [(120, 140), (120, 140, 150)])
@pytest.mark.parametrize('space_order', [4, 8])
@pytest.mark.parametrize('kernel', ['centered', 'shifted'])
def test_tti(shape, space_order, kernel):
    """
    This first test compare the solution of the acoustic wave-equation and the
    TTI wave-eqatuon with all anisotropy parametrs to 0. The two solutions should
    be the same.
    """
    if kernel == 'shifted':
        space_order *= 2
    to = 2
    so = space_order // 2 if kernel == 'shifted' else space_order
    nbpml = 10
    origin = [0. for _ in shape]
    spacing = [10. for _ in shape]
    vp = 1.5 * np.ones(shape)

    # Constant model for true velocity
    model = Model(origin=origin, shape=shape, vp=vp,
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

    # Generate a wavefield as initial condition
    source = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range)
    source.coordinates.data[0, :] = np.array(model.domain_size) * .5

    receiver = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=1)

    acoustic = AcousticWaveSolver(model, source=source, receiver=receiver,
                                  time_order=2, space_order=so)
    rec, u1, _ = acoustic.forward(save=False)

    source.data.fill(0.)
    # Solvers
    acoustic = AcousticWaveSolver(model, source=source, receiver=receiver,
                                  time_order=2, space_order=so)

    solver_tti = AnisotropicWaveSolver(model, source=source, receiver=receiver,
                                       time_order=2, space_order=space_order)

    # Create new wavefield object restart forward computation
    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=so)
    u.data[0:3, :] = u1.data[indlast, :]
    acoustic.forward(save=False, u=u, time_M=10, src=source)

    utti = TimeFunction(name='u', grid=model.grid, time_order=to, space_order=so)
    vtti = TimeFunction(name='v', grid=model.grid, time_order=to, space_order=so)

    utti.data[0:to+1, :] = u1.data[indlast[:to+1], :]
    vtti.data[0:to+1, :] = u1.data[indlast[:to+1], :]

    solver_tti.forward(u=utti, v=vtti, kernel=kernel, time_M=10, src=source)

    normal_u = u.data[:]
    normal_utti = .5 * utti.data[:]
    normal_vtti = .5 * vtti.data[:]

    res = linalg.norm((normal_u - normal_utti - normal_vtti).reshape(-1))**2
    res /= np.linalg.norm(normal_u.reshape(-1))**2

    log("Difference between acoustic and TTI with all coefficients to 0 %2.4e" % res)
    assert np.isclose(res, 0.0, atol=1e-4)


@skipif_yask
@pytest.mark.parametrize('shape', [(50, 60), (50, 60, 70)])
def test_tti_staggered(shape):
    spacing = [10. for _ in shape]

    # Model
    model = demo_model('constant-tti', shape=shape, spacing=spacing)

    # Define seismic data and parameters
    f0 = .010
    dt = model.critical_dt
    t0 = 0.0
    tn = 250.0
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Generate a wavefield as initial condition
    source = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range)
    source.coordinates.data[0, :] = np.array(model.domain_size) * .5

    receiver = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=1)

    # Solvers
    solver_tti = AnisotropicWaveSolver(model, source=source, receiver=receiver,
                                       time_order=2, space_order=8)
    solver_tti2 = AnisotropicWaveSolver(model, source=source, receiver=receiver,
                                        time_order=2, space_order=8)

    # Solve
    configuration['dse'] = 'aggressive'
    configuration['dle'] = 'advanced'
    rec1, u1, v1, _ = solver_tti.forward(kernel='staggered')
    configuration['dle'] = 'basic'
    rec2, u2, v2, _ = solver_tti2.forward(kernel='staggered')

    res1 = np.linalg.norm(u1.data.reshape(-1) - u2.data.reshape(-1))
    res2 = np.linalg.norm(v1.data.reshape(-1) - v2.data.reshape(-1))
    log("DSE/DLE introduces error %2.4e, %2.4e in %d dimensions" % (res1, res2,
                                                                    len(shape)))
    assert np.isclose(res1, 0.0, atol=1e-8)
    assert np.isclose(res2, 0.0, atol=1e-8)

import numpy as np
import pytest
from numpy import linalg

from examples.acoustic import AcousticWaveSolver
from examples.seismic import Model, PointSource, Receiver


@pytest.mark.parametrize('space_order', [4])
@pytest.mark.parametrize('time_order', [2])
@pytest.mark.parametrize('dimensions', [(70, 80)])
def test_gradient(dimensions, time_order, space_order):
    nbpml = 40

    if len(dimensions) == 2:
        # Dimensions in 2D are (x, z)
        origin = (0., 0.)
        spacing = (10., 10.)

        # True velocity
        true_vp = np.ones(dimensions) + .5
        true_vp[:, int(dimensions[1] / 2):] = 2

        # Source location
        location = np.zeros((1, 2))
        location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
        location[0, 1] = origin[1] + 2 * spacing[1]

        # Receiver coordinates
        receiver_coords = np.zeros((101, 2))
        receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                            dimensions[0] * spacing[0], num=101)
        receiver_coords[:, 1] = location[0, 1]

    elif len(dimensions) == 3:
        # Dimensions in 3D are (x, y, z)
        origin = (0., 0., 0.)
        spacing = (15., 15., 15.)

        # True velocity
        true_vp = np.ones(dimensions) + .5
        true_vp[:, :, int(dimensions[2] / 2):] = 2

        # Source location
        location = np.zeros((1, 3))
        location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
        location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
        location[0, 2] = origin[1] + 2 * spacing[2]

        # Receiver coordinates
        receiver_coords = np.zeros((101, 3))
        receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                            dimensions[0] * spacing[0], num=101)
        receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
        receiver_coords[:, 2] = location[0, 2]

    # velocity models
    def smooth10(vel):
        out = np.zeros(dimensions)
        out[:] = vel[:]
        for a in range(5, dimensions[-1]-6):
            if len(dimensions) == 2:
                out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
            else:
                out[:, :, a] = np.sum(vel[:, :, a - 5:a + 5], axis=2) / 10
        return out

    # Smooth velocity
    initial_vp = smooth10(true_vp)
    m0 = initial_vp**-2

    # Model perturbation
    dm = true_vp**-2 - initial_vp**-2
    model = Model(origin, spacing, dimensions, true_vp, nbpml=nbpml)
    model0 = Model(origin, spacing, dimensions, initial_vp, nbpml=nbpml)

    f0 = .010
    dt = model.critical_dt
    if time_order == 4:
        dt *= 1.73
    t0 = 0.0
    tn = 750.0
    nt = int(1+(tn-t0)/dt)

    # Set up the source as Ricker wavelet for f0
    def source(t, f0):
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)

    # Source geometry
    time_series = np.zeros((nt, 1))
    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

    # Define source and receivers and create acoustic wave solver
    src = PointSource(name='src', data=time_series, coordinates=location)
    rec = Receiver(name='rec', ntime=nt, coordinates=receiver_coords)
    wave = AcousticWaveSolver(model, source=src, receiver=rec,
                              time_order=time_order, space_order=space_order)

    # Compute receiver data for the true velocity
    wave.forward(rec=rec, m=model.m)

    # Compute receiver data and full wavefield for the smooth velocity
    rec0, u0, _ = wave.forward(m=model0.m, save=True)

    # Objective function value
    F0 = .5*linalg.norm(rec0.data - rec.data)**2

    # Gradient: <J^T \delta d, dm>
    gradient, _ = wave.gradient(rec0.data - rec.data, u0, m=model0.m)
    G = np.dot(gradient.data.reshape(-1), model.pad(dm).reshape(-1))

    # FWI Gradient test
    H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
    error1 = np.zeros(7)
    error2 = np.zeros(7)
    for i in range(0, 7):
        # Add the perturbation to the model
        model0.m.data[:] = model0.pad(m0 + H[i] * dm)
        # Data for the new model
        d = wave.forward(m=model0.m)[0]
        # First order error Phi(m0+dm) - Phi(m0)
        error1[i] = np.absolute(.5*linalg.norm(d.data - rec.data)**2 - F0)
        # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
        error2[i] = np.absolute(.5*linalg.norm(d.data - rec.data)**2 - F0 - H[i] * G)
        # print(F0, .5*linalg.norm(d - rec)**2, error1[i], H[i] *G, error2[i])
        # print('For h = ', H[i], '\nFirst order errors is : ', error1[i],
        #       '\nSecond order errors is ', error2[i])

    hh = np.zeros(7)
    for i in range(0, 7):
        hh[i] = H[i] * H[i]

    # Test slope of the  tests
    p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
    p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
    print(p1)
    print(p2)
    assert np.isclose(p1[0], 1.0, rtol=0.1)
    assert np.isclose(p2[0], 2.0, rtol=0.1)


if __name__ == "__main__":
    test_gradient(dimensions=(60, 70), time_order=2, space_order=4)

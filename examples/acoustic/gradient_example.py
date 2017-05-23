import numpy as np
from examples.seismic import Model
from devito import time, TimeData, DenseData
from examples.acoustic import AcousticWaveSolver
from examples.seismic.source import PointSource, Receiver
from numpy import linalg


def smooth10(vel, dimensions):
    out = np.zeros(dimensions)
    out[:] = vel[:]
    for a in range(5, dimensions[-1]-6):
        if len(dimensions) == 2:
            out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
        else:
            out[:, :, a] = np.sum(vel[:, :, a - 5:a + 5], axis=2) / 10
    return out


def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1-2.*r**2)*np.exp(-r**2)


def run(dimensions=(50, 50, 50), spacing=(15.0, 15.0, 15.0), tn=750.0,
        time_order=2, space_order=4, nbpml=40, dse='noop', dle='noop'):
    ndim = len(dimensions)
    origin = tuple([0.] * ndim)
    f0 = .010

    t0 = 0.0
    nsrc = 1
    nrec = 101
    # True velocity
    true_vp = np.ones(dimensions) + .5
    true_vp[:, :, int(dimensions[2] / 2):] = 2.

    # Smooth velocity - we use this as our initial m
    initial_vp = smooth10(true_vp, dimensions)
    m0 = initial_vp**-2
    # Model perturbation
    dm = true_vp**-2 - initial_vp**-2
    model = Model(origin, spacing, true_vp.shape, true_vp, nbpml=nbpml)
    model0 = Model(origin, spacing, true_vp.shape, initial_vp, nbpml=nbpml)
    dt = model.critical_dt
    if time_order == 4:
        dt *= 1.73
    dtype = model.dtype
    h = model.get_spacing()
    nt = int(1+(tn-t0)/dt)

    # Source geometry
    time_series = np.zeros((nt, nsrc))
    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

    location = np.zeros((nsrc, 3))
    location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
    location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    location[0, 2] = origin[1] + 2 * spacing[2]

    # Receiver geometry
    receiver_coords = np.zeros((nrec, 3))
    receiver_coords[:, 0] = np.linspace(0, origin[0] + dimensions[0] * spacing[0],
                                        num=101)
    receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    receiver_coords[:, 2] = location[0, 2]

    # Create source symbol
    src = PointSource(name="src", data=time_series, coordinates=location)

    # Receiver for true model
    recT = Receiver(name="rec", ntime=nt, coordinates=receiver_coords)
    # Receiver for smoothed model
    rec0 = Receiver(name="rec", ntime=nt, coordinates=receiver_coords)

    # Create wave solver from model, source and receiver definitions
    solver = AcousticWaveSolver(model, source=src, receiver=recT,
                                time_order=time_order,
                                space_order=space_order)

    # Calculate receiver data for true velocity
    recT, _, _ = solver.forward(rec=recT, m=model.m, save=False)

    # Smooth velocity
    # This is the pass that needs checkpointing <----
    rec0, u0, _ = solver.forward(rec=rec0, m=model0.m, save=True)

    # Objective function value
    F0 = .5*linalg.norm(rec0.data - recT.data)**2
    # Receiver for Gradient
    # Confusing nomenclature because this is actually the source for the adjoint
    # mode
    rec_g = Receiver(name="rec", data=rec0.data - recT.data,
                     coordinates=receiver_coords)

    # Gradient symbol
    grad = DenseData(name="grad", shape=model.shape_domain, dtype=model.dtype)
    # Apply the gradient operator to calculate the gradient
    # This is the pass that requires the checkpointed data
    grad, _ = solver.gradient(rec_g, u0, grad=grad)
    # The result is in grad
    gradient = grad.data

    # <J^T \delta d, dm>
    G = np.dot(gradient.reshape(-1), model.pad(dm).reshape(-1))
    # FWI Gradient test
    H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
    error1 = np.zeros(7)
    error2 = np.zeros(7)

    for i in range(0, 7):
        # Add the perturbation to the model
        model0.m.data[:] = model.pad(m0 + H[i] * dm)
        # Receiver data for the new model
        d, _, _ = solver.forward(m=model0.m, save=False)
        # First order error Phi(m0+dm) - Phi(m0)
        error1[i] = np.absolute(.5*linalg.norm(d.data - recT.data)**2 - F0)
        # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
        error2[i] = np.absolute(.5*linalg.norm(d.data - recT.data)**2 - F0 - H[i] * G)

    # Test slope of the  tests
    p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
    p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
    print(p1, p2)
    assert np.isclose(p1[0], 1.0, rtol=0.1)
    assert np.isclose(p2[0], 2.0, rtol=0.1)


if __name__ == "__main__":
    run(dimensions=(60, 70, 80), time_order=2, space_order=4)

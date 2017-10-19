import numpy as np
from numpy import linalg

from devito import TimeFunction, Function
from examples.seismic import Receiver, RickerSource, demo_model
from examples.seismic.acoustic import ForwardOperator, GradientOperator, smooth10


def run(shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0), tn=500., time_order=2,
        space_order=4, nbpml=10):
    nrec = shape[0]
    model = demo_model('layers-isotropic', shape=shape, spacing=spacing, nbpml=nbpml)
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time = np.linspace(t0, tn, nt)  # Discretized time axis

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, just below surface)
    # We need two receivers - one for the true (verification) run
    rec_t = Receiver(name='rec_t', grid=model.grid, ntime=nt, npoint=nrec)
    rec_t.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec_t.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # and the other for the smoothed run
    rec_s = Receiver(name='rec_s', grid=model.grid, ntime=nt, npoint=nrec,
                     coordinates=rec_t.coordinates.data)

    # Create the forward wavefield to use (only 3 timesteps)
    # Once checkpointing is in, this will be the only wavefield we need
    u_nosave = TimeFunction(name="u", grid=model.grid, time_order=time_order,
                            space_order=space_order, save=False, dtype=model.dtype)

    # Forward wavefield where all timesteps are stored
    # With checkpointing this should go away <----
    u_save = TimeFunction(name="u", grid=model.grid, time_dim=nt,
                          time_order=time_order, space_order=space_order, save=True,
                          dtype=model.dtype)

    # Forward Operators - one with save = True and one with save = False
    fw = ForwardOperator(model, src, rec_t, time_order=time_order,
                         spc_order=space_order, save=True)

    fw_nosave = ForwardOperator(model, src, rec_t, time_order=time_order,
                                spc_order=space_order, save=False)

    # Calculate receiver data for true velocity
    fw_nosave.apply(u=u_nosave, rec=rec_t, src=src, dt=dt)

    m0 = smooth10(model.m.data, model.shape_domain)
    dm = np.float32(model.m.data - m0)

    # Smooth velocity
    # This is the pass that needs checkpointing <----
    fw.apply(u=u_save, rec=rec_s, m=m0, src=src, dt=dt)

    # Objective function value
    F0 = .5*linalg.norm(rec_s.data - rec_t.data)**2
    # Receiver for Gradient
    rec_g = Receiver(name="rec", coordinates=rec_s.coordinates.data, grid=model.grid,
                     data=rec_s.data - rec_t.data, dt=dt)
    # Gradient symbol
    grad = Function(name="grad", grid=model.grid, dtype=model.dtype)
    # Reusing u_nosave from above as the adjoint wavefield since it is a temp var anyway
    gradop = GradientOperator(model, src, rec_g, time_order=time_order,
                              spc_order=space_order)

    # Clear the wavefield variable to reuse it
    # This time it represents the adjoint field
    u_nosave.data.fill(0)
    # Apply the gradient operator to calculate the gradient
    # This is the pass that requires the checkpointed data
    gradop.apply(u=u_save, v=u_nosave, m=m0, rec=rec_g, grad=grad, dt=dt)
    # The result is in grad
    gradient = grad.data

    # <J^T \delta d, dm>
    G = np.dot(gradient.reshape(-1), dm.reshape(-1))
    # FWI Gradient test
    H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
    error1 = np.zeros(7)
    error2 = np.zeros(7)

    for i in range(0, 7):
        # Add the perturbation to the model
        mloc = m0 + H[i] * dm
        # Set field to zero (we're re-using it)
        u_nosave.data.fill(0)
        # Receiver data for the new model
        # Results will be in rec_s
        fw_nosave.apply(u=u_nosave, rec=rec_s, m=mloc, src=src, dt=dt)
        d = rec_s.data
        # First order error Phi(m0+dm) - Phi(m0)
        error1[i] = np.absolute(.5*linalg.norm(d - rec_t.data)**2 - F0)
        # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
        error2[i] = np.absolute(.5*linalg.norm(d - rec_t.data)**2 - F0 - H[i] * G)

    # Test slope of the  tests
    p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
    p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
    assert np.isclose(p1[0], 1.0, rtol=0.1)
    assert np.isclose(p2[0], 2.0, rtol=0.1)


if __name__ == "__main__":
    run(shape=(150, 150), spacing=(15.0, 15.0), tn=750.0, time_order=2, space_order=4)

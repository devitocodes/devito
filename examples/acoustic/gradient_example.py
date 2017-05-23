import numpy as np
from examples.seismic import Model
from devito import Dimension, time, TimeData, DenseData
from examples.source_type import SourceLike
from examples.acoustic.fwi_operators import ForwardOperator, GradientOperator
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
        time_order=2, space_order=4, nbpml=40, dse='noop', dle='noop',
        autotuning=False, compiler=None, cache_blocking=None):
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
    p_src = Dimension('p_src', size=nsrc)
    src = SourceLike(name="src", dimensions=[time, p_src], npoint=nsrc, nt=nt,
                     dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                     coordinates=location)
    src.data[:] = time_series

    # Create receiver symbol
    p_rec = Dimension('p_rec', size=nrec)

    # Receiver for true model
    rec_t = SourceLike(name="rec_t", dimensions=[time, p_rec], npoint=nrec, nt=nt, dt=dt,
                       h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                       coordinates=receiver_coords)
    # Receiver for smoothed model
    rec_s = SourceLike(name="rec_s", dimensions=[time, p_rec], npoint=nrec, nt=nt, dt=dt,
                       h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                       coordinates=receiver_coords)

    # Create the forward wavefield to use (only 3 timesteps)
    # Once checkpointing is in, this will be the only wavefield we need
    u_nosave = TimeData(name="u_ns", shape=model.shape_domain, time_dim=nt,
                        time_order=time_order, space_order=space_order, save=False,
                        dtype=model.dtype)

    # Forward wavefield where all timesteps are stored
    # With checkpointing this should go away <----
    u_save = TimeData(name="u_s", shape=model.shape_domain, time_dim=nt,
                      time_order=time_order, space_order=space_order, save=True,
                      dtype=model.dtype)

    # Forward Operators - one with save = True and one with save = False
    fw = ForwardOperator(model, u_save, src, rec_t, time_order=time_order,
                         spc_order=space_order, save=True, cache_blocking=cache_blocking,
                         dse=dse, dle=dle, compiler=compiler, profile=True)

    fw_nosave = ForwardOperator(model, u_nosave, src, rec_t, time_order=time_order,
                                spc_order=space_order, save=False,
                                cache_blocking=cache_blocking, dse=dse, dle=dle,
                                compiler=compiler, profile=True)

    # Calculate receiver data for true velocity
    fw_nosave.apply(autotuning=autotuning, rec_t=rec_t)

    # Change to the smooth velocity
    model.m.data[:] = model.pad(1 / initial_vp ** 2)

    # Smooth velocity
    # This is the pass that needs checkpointing <----
    fw.apply(autotuning=autotuning, rec_t=rec_s)

    # Objective function value
    F0 = .5*linalg.norm(rec_s.data - rec_t.data)**2

    # Receiver for Gradient
    # Confusing nomenclature because this is actually the source for the adjoint
    # mode
    rec_g = SourceLike(name="rec_g", dimensions=[time, p_rec], npoint=nrec, nt=nt, dt=dt,
                       h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                       coordinates=receiver_coords)
    rec_g.data[:] = rec_s.data - rec_t.data

    # Gradient symbol
    grad = DenseData(name="grad", shape=model.shape_domain, dtype=model.dtype)
    # Reusing u_nosave from above as the adjoint wavefield since it is a temp var anyway
    gradop = GradientOperator(model, u_nosave, grad, rec_g, u_save, time_order=time_order,
                              spc_order=space_order, cache_blocking=cache_blocking,
                              dse=dse, dle=dle, compiler=compiler, profile=True)

    # Clear the wavefield variable to reuse it
    # This time it represents the adjoint field
    u_nosave.data.fill(0)
    # Apply the gradient operator to calculate the gradient
    # This is the pass that requires the checkpointed data
    gradop.apply()
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
        model.m.data[:] = model.pad(m0 + H[i] * dm)
        # Set field to zero (we're re-using it)
        u_nosave.data.fill(0)
        # Receiver data for the new model
        # Results will be in rec_s
        fw_nosave.apply(rec_t=rec_s)
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
    run(dimensions=(60, 70, 80), time_order=2, space_order=4)

import numpy as np
from numpy import linalg
from math import floor
from devito import Function, TimeFunction
from examples.seismic import Receiver, demo_model, RickerSource
from examples.seismic.acoustic import ForwardOperator, GradientOperator, smooth10
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver


def setup(shape, tn, spacing, time_order, space_order, nbpml):
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

    # Receiver for Gradient run
    rec_g = Receiver(name="rec_g", grid=model.grid, ntime=nt,
                     coordinates=rec_s.coordinates.data)

    # Create the forward wavefield to use (only 3 timesteps)
    u = TimeFunction(name="u", grid=model.grid, time_order=time_order,
                     space_order=space_order, save=False)

    v = TimeFunction(name="v", grid=model.grid, time_order=time_order,
                     space_order=space_order, save=False)

    # Forward Operator
    fw = ForwardOperator(model, src, rec_t, time_order=time_order,
                         spc_order=space_order, save=False)

    # Gradient symbol
    grad = Function(name="grad", grid=model.grid)

    gradop = GradientOperator(model, src, rec_g, time_order=time_order,
                              spc_order=space_order, save=False)
    # Calculate receiver data for true velocity
    fw.apply(u=u, rec=rec_t, src=src, dt=dt)
    u.data[:] = 0
    m0 = smooth10(model.m.data, model.shape_domain)
    dm = np.float32(model.m.data - m0)
    return fw, gradop, u, rec_s, m0, src, rec_g, v, grad, rec_t, dm, nt, dt


def gradient(fw, gradop, u, maxmem, rec_s, m0, src, rec_g, v, grad, rec_t, nt, dt):
    cp = DevitoCheckpoint([u])
    n_checkpoints = None
    if maxmem is not None:
        n_checkpoints = int(floor(maxmem*10**6/(cp.size*u.data.itemsize)))

    wrap_fw = CheckpointOperator(fw, {'u': u, 'rec': rec_s, 'm': m0, 'src': src,
                                      'dt': dt})
    wrap_rev = CheckpointOperator(gradop, {'u': u, 'v': v, 'm': m0, 'rec': rec_g,
                                           'grad': grad, 'dt': dt})
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-2)

    wrp.apply_forward()

    rec_g.data[:] = rec_s.data[:] - rec_t.data[:]

    wrp.apply_reverse()

    # The result is in grad
    return grad.data


def verify(gradient, dm, m0, u, rec_s, fw, src, rec_t, dt):
    # Objective function value
    F0 = .5*linalg.norm(rec_s.data - rec_t.data)**2
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
        u.data.fill(0)
        rec_s.data.fill(0)
        # Receiver data for the new model
        # Results will be in rec_s
        fw.apply(u=u, rec=rec_s, m=mloc, src=src, dt=dt)
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


class CheckpointedGradientExample(object):
    def __init__(self, dimensions=(150, 150), tn=None, spacing=None, time_order=2,
                 space_order=4, nbpml=10):
        (self.fw, self.gradop, self.u, self.rec_s, self.m0, self.src, self.rec_g, self.v,
         self.grad, self.rec_t, self.dm, self.nt, self.dt) = setup(dimensions, tn,
                                                                   spacing, time_order,
                                                                   space_order, nbpml)

    def do_gradient(self, maxmem):
        return gradient(self.fw, self.gradop, self.u, maxmem, self.rec_s, self.m0,
                        self.src, self.rec_g, self.v, self.grad, self.rec_t, self.nt,
                        self.dt)

    def do_verify(self, grad):
        verify(grad, self.dm, self.m0, self.v, self.rec_s, self.fw, self.src, self.rec_t,
               self.dt)


def run(shape=(150, 150), tn=None, spacing=None, time_order=2, space_order=4, nbpml=10,
        maxmem=None):
    ex = CheckpointedGradientExample(shape, tn, spacing, time_order, space_order, nbpml)
    grad = ex.do_gradient(maxmem)
    ex.do_verify(grad)


if __name__ == "__main__":
    run(shape=(150, 150), spacing=(15.0, 15.0), tn=750.0, time_order=2, space_order=4)

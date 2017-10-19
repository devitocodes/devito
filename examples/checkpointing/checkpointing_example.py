import numpy as np
from numpy import linalg
from math import floor
from devito import TimeData, DenseData
from examples.seismic import Model, PointSource, Receiver
from examples.seismic.acoustic import ForwardOperator, GradientOperator
from examples.checkpointing.checkpoint import DevitoCheckpoint, DevitoOperator
from pyrevolve import Revolver


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


def setup(dimensions, tn, spacing, time_order, space_order, nbpml, dse, dle):
    ndim = len(dimensions)
    origin = tuple([0.] * ndim)
    if spacing is None:
        spacing = tuple([15.] * ndim)
    
    if tn is None:
        vmin = 1.5
        tn = 2*dimensions[0]*spacing[0]/vmin

    f0 = .010
    t0 = 0.0
    # True velocity
    true_vp = np.ones(dimensions) + .5
    true_vp[..., int(dimensions[-1] / 2):] = 2.

    # Smooth velocity - we use this as our initial m
    initial_vp = smooth10(true_vp, dimensions)
    # Model perturbation
    model = Model(origin, spacing, true_vp.shape, true_vp, nbpml=nbpml)
    m0 = np.float32(model.pad(initial_vp**-2))
    dm = np.float32(model.m.data - m0)
    dt = model.critical_dt
    if time_order == 4:
        dt *= 1.73

    nt = int(1+(tn-t0)/dt)

    # Source geometry
    time_series = np.zeros((nt, 1), dtype=np.float32)
    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

    # Source location
    location = np.zeros((1, ndim), dtype=np.float32)
    location[0, :-1] = [origin[i] + dimensions[i] * spacing[i] * .5
                        for i in range(ndim-1)]
    location[0, -1] = origin[-1] + 2 * spacing[-1]
    # Receivers locations
    receiver_coords = np.zeros((dimensions[0], ndim), dtype=np.float32)
    receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                        (dimensions[0]-1) * spacing[0],
                                        num=dimensions[0])
    receiver_coords[:, 1:] = location[0, 1:]

    # Create source symbol
    src = PointSource(name="src", data=time_series, coordinates=location)

    # Receiver for true model
    rec_t = Receiver(name="rec_t", ntime=nt, coordinates=receiver_coords)
    # Receiver for smoothed model
    rec_s = Receiver(name="rec_s", ntime=nt, coordinates=receiver_coords)

    # Receiver for Gradient
    # Confusing nomenclature because this is actually the source for the adjoint
    # mode
    rec_g = Receiver(name="rec_g", ntime=nt, coordinates=rec_s.coordinates.data)

    # Create the forward wavefield to use (only 3 timesteps)
    u = TimeData(name="u", shape=model.shape_domain, time_order=time_order,
                 space_order=space_order, save=False, dtype=model.dtype)

    v = TimeData(name="v", shape=model.shape_domain, time_order=time_order,
                 space_order=space_order, save=False, dtype=model.dtype)

    # Forward Operators - one with save = True and one with save = False
    fw = ForwardOperator(model, src, rec_t, time_order=time_order,
                                spc_order=space_order, save=False, dse=dse, dle=dle)

     # Gradient symbol
    grad = DenseData(name="grad", shape=model.shape_domain, dtype=model.dtype)
    
    gradop = GradientOperator(model, src, rec_g, time_order=time_order,
                              spc_order=space_order, save=False, dse=dse, dle=dle)

    # Calculate receiver data for true velocity
    fw.apply(u=u, rec=rec_t, src=src)
    u.data[:] = 0
    return fw, gradop, u, rec_s, m0, src, rec_g, v, grad, rec_t, dm, nt


def gradient(fw, gradop, u, maxmem, rec_s, m0, src, rec_g, v, grad, rec_t, nt):
    cp = DevitoCheckpoint([u])
    n_checkpoints = None
    print(maxmem)
    if maxmem is not None:
        n_checkpoints = int(floor(maxmem*10**6/(cp.size*u.data.itemsize)))
        print("Checkpoints: %d * %d" % (n_checkpoints, cp.size))
    print("Timesteps: %d"%nt)
    wrap_fw = DevitoOperator(fw, {'u': u, 'rec': rec_s, 'm': m0, 'src': src}, {'t_start': 't_s', 't_end': 't_e'}, [0, 2])
    wrap_rev = DevitoOperator(gradop, {'u':u, 'v': v, 'm': m0, 'rec': rec_g,'grad':grad}, {'t_start': 't_s', 't_end': 't_e'}, [0, 2, 1])
    wrp = Revolver(cp, wrap_fw, wrap_rev, nt-2, n_checkpoints)
    
    # Smooth velocity
    # This is the pass that needs checkpointing <----
    # fw.apply(u=u, rec=rec_s, m=m0, src=src)

    #fw.apply(u=u1, rec=rec_s, m=m0, src=src)
    #rec_s_backup = np.copy(rec_s.data)
    #rec_s.data[:] = 0
    #u.data[:] = 0
    wrp.apply_forward()
    #assert(np.allclose(u.data, u1.data))
    #assert(np.allclose(rec_s.data, rec_s_backup))
    print(np.linalg.norm(rec_s.data))
    rec_g.data[:] = rec_s.data[:] - rec_t.data[:]

    # Apply the gradient operator to calculate the gradient
    # This is the pass that requires the checkpointed data
    # gradop.apply(u=u, v=v, m=m0, rec=rec_g, grad=grad)
    wrp.apply_reverse()
    
    # The result is in grad
    return grad.data

def verify(gradient, dm, m0, u, rec_s, fw, src, rec_t):
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
        fw.apply(u=u, rec=rec_s, m=mloc, src=src)
        d = rec_s.data
        # First order error Phi(m0+dm) - Phi(m0)
        error1[i] = np.absolute(.5*linalg.norm(d - rec_t.data)**2 - F0)
        # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
        error2[i] = np.absolute(.5*linalg.norm(d - rec_t.data)**2 - F0 - H[i] * G)

    # Test slope of the  tests
    p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
    p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
    print(p1)
    print(p2)
    assert np.isclose(p1[0], 1.0, rtol=0.1)
    assert np.isclose(p2[0], 2.0, rtol=0.1)

class CheckpointedGradientExample(object):
    def __init__(self, dimensions=(50, 50, 50), tn=None, spacing=None, 
        time_order=2, space_order=4, nbpml=40, dse='advanced', dle='advanced'):
        self.fw, self.gradop, self.u, self.rec_s, self.m0, self.src, self.rec_g, self.v, self.grad, self.rec_t, self.dm, self.nt = setup(dimensions, tn, spacing, time_order, space_order, nbpml, dse, dle)

    def do_gradient(self, maxmem):
        return gradient(self.fw, self.gradop, self.u, maxmem, self.rec_s, self.m0, self.src, self.rec_g, self.v, self.grad, self.rec_t, self.nt)

    def do_verify(self, grad):
        verify(grad, self.dm, self.m0, self.v, self.rec_s, self.fw, self.src, self.rec_t)

def run(dimensions=(50, 50, 50), tn=None, spacing=None, autotune=False, 
        time_order=2, space_order=4, nbpml=40, maxmem=None, dse='advanced', dle='advanced'):
    ex = CheckpointedGradientExample(dimensions, tn, spacing, time_order, space_order, nbpml, dse, dle)
    grad = ex.do_gradient(maxmem)
    ex.do_verify(grad)

if __name__ == "__main__":
    run(dimensions=(60, 70), time_order=2, space_order=4)

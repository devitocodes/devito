import numpy as np
from math import floor
from containers import IShot, IGrid
from devito.interfaces import DenseData, TimeData
from fwi_operators import SourceLike, ForwardOperator, AdjointOperator, GradientOperator, BornOperator

dimensions = (50, 50, 50)
model = IGrid()
model0 = IGrid()
model1 = IGrid()
model.shape = dimensions
model0.shape = dimensions
model1.shape = dimensions
origin = (0., 0.)
spacing = (25., 25.)
dtype = np.float32
t_order = 2
spc_order = 2


# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape)
    out[:, :] = vel[:, :]
    nx = shape[0]
    for a in range(5, nx-6):
        out[a, :] = np.sum(vel[a - 5:a + 5, :], axis=0) / 10
    return out


# True velocity
true_vp = np.ones(dimensions) + 2.0
true_vp[floor(dimensions[0] / 2):floor(dimensions[0]), :] = 4.5

# Smooth velocity
initial_vp = smooth10(true_vp, dimensions)

nbpml = 40
dm_orig = true_vp**-2 - initial_vp**-2
pad_list = []
for dim_index in range(len(dimensions)):
    pad_list.append((nbpml, nbpml))

dm_orig = np.pad(dm_orig, tuple(pad_list), 'edge')


def create_dm(dm):
    np.copyto(dm, dm_orig)

dv = -true_vp + initial_vp

model.create_model(origin, spacing, true_vp)
model0.create_model(origin, spacing, initial_vp)
model1.create_model(origin, spacing, initial_vp)

# Define seismic data.
data = IShot()

f0 = .010
dt = model.get_critical_dt()
t0 = 0.0
tn = 100.0
nt = int(1+(tn-t0)/dt)
h = model.get_spacing()
model.vp = np.pad(model.vp, tuple(pad_list), 'edge')
data.reinterpolate(dt)
model.set_origin(nbpml)


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1-2.*r**2)*np.exp(-r**2)


time_series = source(np.linspace(t0, tn, nt), f0)
location = (origin[0] + dimensions[0] * spacing[0] * 0.5, 500,
            origin[1] + 2 * spacing[1])
data.set_source(time_series, dt, location)

receiver_coords = np.zeros((101, 3))
receiver_coords[:, 0] = np.linspace(50, 950, num=101)
receiver_coords[:, 1] = 500
receiver_coords[:, 2] = location[2]
data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 101)


def damp_boundary(damp, h, nbpml):
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40 * h)
    num_dim = len(damp.shape)
    for i in range(nbpml):
        pos = np.abs((nbpml-i)/float(nbpml))
        val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
        if num_dim == 2:
            damp[i, :] += val
            damp[-(i + 1), :] += val
            damp[:, i] += val
            damp[:, -(i + 1)] += val
        else:
            damp[i, :, :] += val
            damp[-(i + 1), :, :] += val
            damp[:, i, :] += val
            damp[:, -(i + 1), :] += val
            damp[:, :, i] += val
            damp[:, :, -(i + 1)] += val

nrec, _ = data.traces.shape
m = DenseData("m", model.vp.shape, dtype)
m.data[:] = model.vp**(-2)
u = TimeData("u", m.shape, nt, time_order=t_order, save=False, dtype=m.dtype)
v = TimeData("v", m.shape, nt, time_order=t_order, save=False, dtype=m.dtype)

damp = DenseData("damp", model.vp.shape, m.dtype)
damp_boundary(damp.data, h, nbpml)
src = SourceLike("src", 1, nt, dt, h, np.array(data.source_coords, dtype=dtype)[np.newaxis, :], len(dimensions), dtype, nbpml)
rec = SourceLike("rec", nrec, nt, dt, h, data.receiver_coords, len(dimensions), dtype, nbpml)
src.data[:] = data.get_source()[:, np.newaxis]
print("Preparing Forward")
forward_op = ForwardOperator(m, src, damp, rec, u, t_order, spc_order)
print("Applying")
forward_op.apply()
srca = SourceLike("srca", 1, nt, dt, h, np.array(data.source_coords, dtype=dtype)[np.newaxis, :], len(dimensions), dtype, nbpml)
print("Preparing adjoint")
adjoint_op = AdjointOperator(m, rec, damp, srca, t_order, spc_order)
print("Applying")
adjoint_op.apply()
print("Preparing Gradient")
gradient_op = GradientOperator(u, m, rec, damp, t_order, spc_order)
print("Applying")
gradient_op.apply()
print("Preparing Born")
dm = DenseData("dm", model.vp.shape, dtype)
create_dm(dm.data)
born_op = BornOperator(dm, m, src, damp, rec, t_order, spc_order)
print("Applying")
born_op.apply()

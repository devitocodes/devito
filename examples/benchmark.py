from AcousticWave2D_codegen import AcousticWave2D_cg
import numpy as np
from AcousticWave2D import AcousticWave2D
import time
from math import floor
from terminaltables import AsciiTable
from containers import IShot, IGrid


dimensions = (100, 100)
model = IGrid()
model0 = IGrid()
model1 = IGrid()
model.shape = dimensions
model0.shape = dimensions
model1.shape = dimensions
origin = (0., 0.)
spacing = (25., 25)


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
tn = 1000.0
nt = int(1+(tn-t0)/dt)


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1-2.*r**2)*np.exp(-r**2)


time_series = source(np.linspace(t0, tn, nt), f0)
location = (origin[0] + dimensions[0] * spacing[0] * 0.5, 0,
            origin[1] + 2 * spacing[1])
data.set_source(time_series, dt, location)

receiver_coords = np.zeros((30, 3))
receiver_coords[:, 0] = np.linspace(50, 950, num=30)
receiver_coords[:, 1] = 0.0
receiver_coords[:, 2] = location[2]
data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 30)
# A Forward propagation example
jit_obj = AcousticWave2D_cg(model1, data, create_dm, nbpml=nbpml, t_order=2, s_order=2)
python_obj = AcousticWave2D(model0, data, nbpml=nbpml)


print "Forward propagation"
print "Starting codegen version"
start = time.clock()
(recg, ug) = jit_obj.Forward()
end = time.clock()
cg_time = end-start
norm_recg = np.linalg.norm(recg)
norm_ug = np.linalg.norm(ug)
print "Starting python lambdified version"
start = time.clock()
(rect, ut) = python_obj.Forward()
end = time.clock()
python_time = end-start
norm_rect = np.linalg.norm(rect)
norm_ut = np.linalg.norm(ut)


table_data = [
    ['', 'Time', 'L2Norm(u)', 'L2Norm(rec)'],
    ['Python lambdified', str(python_time), str(norm_ut), str(norm_rect)],
    ['Codegen', str(cg_time), str(norm_ug), str(norm_recg)]
]
table = AsciiTable(table_data)
print table.table

print "Adjoint propagation"
print "Starting python lambdified version"
start = time.clock()
(srca_t, v_t) = python_obj.Adjoint(recg)
end = time.clock()
python_time = end-start
norm_srct = np.linalg.norm(srca_t)
norm_vt = np.linalg.norm(v_t)

print "Starting codegen version"

start = time.clock()
(srca_g, v_g) = jit_obj.Adjoint(recg)
end = time.clock()
cg_time = end-start
norm_srcg = np.linalg.norm(srca_g)
norm_vg = np.linalg.norm(v_g)

table_data = [
    ['', 'Time', 'L2Norm(v)', 'L2Norm(src)'],
    ['Python lambdified', str(python_time), str(norm_vt), str(norm_srct)],
    ['Codegen', str(cg_time), str(norm_vg), str(norm_srcg)]
]
table = AsciiTable(table_data)
print table.table

print "Gradient propagation"
print "Starting python lambdified version"
start = time.clock()
grad_t = python_obj.Gradient(recg, ug)
end = time.clock()
python_time = end-start
norm_gradt = np.linalg.norm(grad_t)

print "Starting codegen version"

start = time.clock()
grad_g = jit_obj.Gradient(recg, ug)
end = time.clock()
cg_time = end-start
norm_gradg = np.linalg.norm(grad_g)

table_data = [
    ['', 'Time', 'L2Norm(grad)'],
    ['Python lambdified', str(python_time), str(norm_gradt)],
    ['Codegen', str(cg_time), str(norm_gradg)]
]
table = AsciiTable(table_data)
print table.table

print "Born"
print "Starting python lambdified version"
start = time.clock()
rec_t = python_obj.Born(dm_orig)
end = time.clock()
python_time = end-start
norm_rect = np.linalg.norm(rec_t)

print "Starting codegen version"

start = time.clock()
rec_g = jit_obj.Born()
end = time.clock()
cg_time = end-start
norm_recg = np.linalg.norm(rec_g)
table_data = [
    ['', 'Time', 'L2Norm(rec)'],
    ['Python lambdified', str(python_time), str(norm_rect)],
    ['Codegen', str(cg_time), str(norm_recg)]
]
table = AsciiTable(table_data)
print table.table
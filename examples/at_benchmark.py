from AcousticWave2D_codegen import AcousticWave2D_cg
import numpy as np
import time
from math import floor
from containers import IShot, IGrid


# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape)
    out[:, :] = vel[:, :]
    nx = shape[0]
    for a in range(5, nx-6):
        out[a, :] = np.sum(vel[a - 5:a + 5, :], axis=0) / 10
    return out


def create_dm(dm):
    np.copyto(dm, dm_orig)


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1. / f0))
    return (1 - 2. * r ** 2) * np.exp(-r ** 2)


dimensions = (100, 100, 100)
model = IGrid()
model.shape = dimensions
origin = (0., 0.)
spacing = (25., 25)

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

dv = -true_vp + initial_vp

model.create_model(origin, spacing, true_vp)

# Define seismic data.
data = IShot()

f0 = .010
dt = model.get_critical_dt()
t0 = 0.0
tn = 1000.0
nt = int(1+(tn-t0)/dt)

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

# A Forward trying to find best block sizes using auto tuning
for time_order in xrange(2, 6, 2):
    for space_order in xrange(2, 18, 2):
        new_model = IGrid()
        new_model.shape = dimensions
        new_model.create_model(origin, spacing, initial_vp)

        jit_obj = AcousticWave2D_cg(new_model, data, create_dm, nbpml=nbpml, t_order=time_order,
                                    s_order=space_order, cache_blocking=True, auto_tune=True)

        print "Forward propagation with auto tuning "
        print "Starting codegen version"
        start = time.clock()
        (recg, ug) = jit_obj.Forward()
        end = time.clock()
        cg_time = end-start
        norm_recg = np.linalg.norm(recg)
        norm_ug = np.linalg.norm(ug)

# A Forward using at block sizes which are picked up from auto-tuning report
for time_order in xrange(2, 6, 2):
    for space_order in xrange(2, 18, 2):
        new_model = IGrid()
        new_model.shape = dimensions
        new_model.create_model(origin, spacing, initial_vp)

        jit_obj = AcousticWave2D_cg(new_model, data, create_dm, nbpml=nbpml, t_order=time_order, s_order=space_order,
                                    cache_blocking=True)

        print "Forward propagation"
        print "Starting codegen version"
        start = time.clock()
        (recg, ug) = jit_obj.Forward()
        end = time.clock()
        cg_time = end - start
        norm_recg = np.linalg.norm(recg)
        norm_ug = np.linalg.norm(ug)

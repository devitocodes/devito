from os import path

import numpy as np

from Acoustic_codegen import Acoustic_cg
from containers import IGrid, IShot
from devito.at_controller import AutoTuner
from fwi_operators import ForwardOperator


def create_dm(dm):
    np.copyto(dm, dm_orig)


# Velocity models
def smooth10(vel, shape):
    out = np.ones(shape)
    out[:, :] = vel[:, :]
    nx = shape[0]

    for a in range(5, nx-6):
        out[a, :] = np.sum(vel[a - 5:a + 5, :], axis=0) / 10
    return out


# Set up the source as Ricker wavelet for f0
def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1-2.*r**2)*np.exp(-r**2)

dimensions = (50, 50, 50)
origin = (0., 0.)
spacing = (20., 20.)
dtype = np.float32


# True velocity
true_vp = np.ones(dimensions) + 2.0
true_vp[:, :, int(dimensions[0] / 2):int(dimensions[0])] = 4.5

# Smooth velocity
initial_vp = smooth10(true_vp, dimensions)

dm_orig = true_vp**-2 - initial_vp**-2
dv = -true_vp + initial_vp

model = IGrid()
model.shape = dimensions
model.create_model(origin, spacing, true_vp)

# Define seismic data.
data = IShot()

f0 = .010
dt = model.get_critical_dt()
t0 = 0.0
tn = 250.0
nt = int(1+(tn-t0)/dt)
h = model.get_spacing()
data.reinterpolate(dt)


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

at_report_dir = path.join(path.dirname(path.realpath(__file__)), "At Report")
cache_blocking = [True, True, False]  # not blocking outer most dim

# auto tuning over all space time orders
for time_order in xrange(2, 4, 2):
    for space_order in xrange(2, 16, 2):
        model = IGrid()
        model.shape = dimensions
        model.create_model(origin, spacing, true_vp)

        Acoustic = Acoustic_cg(model, data, dm_orig, None, nbpml=10, t_order=2, s_order=2)
        fw = ForwardOperator(Acoustic.m, Acoustic.src, Acoustic.damp, Acoustic.rec, Acoustic.u,
                             time_order=time_order, spc_order=space_order, cache_blocking=cache_blocking)

        at = AutoTuner(fw, at_report_dir)
        at.auto_tune_blocks(5, 26)

# using auto tuned block sizes
for time_order in xrange(2, 4, 2):
    for space_order in xrange(2, 16, 2):
        model = IGrid()
        model.shape = dimensions
        model.create_model(origin, spacing, true_vp)

        Acoustic = Acoustic_cg(model, data, dm_orig, None, nbpml=10, t_order=2, s_order=2)
        fw = ForwardOperator(Acoustic.m, Acoustic.src, Acoustic.damp, Acoustic.rec, Acoustic.u,
                             time_order=time_order, spc_order=space_order, cache_blocking=cache_blocking,
                             at_report=at_report_dir)
        fw.apply()

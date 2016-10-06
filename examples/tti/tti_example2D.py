import numpy as np

from examples.containers import IGrid, IShot
from examples.tti.TTI_codegen import TTI_cg

dimensions = (150, 150)
model = IGrid()
model.shape = dimensions
origin = (0., 0.)
spacing = (20.0, 20.0)
dtype = np.float32
t_order = 2
spc_order = 4
# True velocity
true_vp = np.ones(dimensions) + 1.0
true_vp[:, int(dimensions[1] / 3):int(2*dimensions[1]/3)] = 3.0
true_vp[:, int(2*dimensions[1] / 3):int(dimensions[1])] = 4.0

model.create_model(origin, spacing, true_vp, 0.1*(true_vp - 2),
                   0.08 * (true_vp - 2), np.pi/5*np.ones(dimensions),
                   0*np.ones(dimensions))

# Define seismic data.
data = IShot()

f0 = .010
dt = model.get_critical_dt()
t0 = 0.0
tn = 2000.0
nt = int(1+(tn-t0)/dt)
h = model.get_spacing()
# Set up the source as Ricker wavelet for f0


def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1-2.*r**2)*np.exp(-r**2)
time_series = source(np.linspace(t0, tn, nt), f0)
location = (origin[0] + dimensions[0] * spacing[0] * 0.5, 40)
data.set_source(time_series, dt, location)
receiver_coords = np.zeros((301, 2))
receiver_coords[:, 0] = np.linspace(50, 2950, num=301)
receiver_coords[:, 1] = 40
data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 301)

TTI = TTI_cg(model, data, None, t_order=2, s_order=spc_order, nbpml=10)
rec, u, v, gflops, oi, timings = TTI.Forward()

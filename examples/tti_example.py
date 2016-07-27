from containers import IShot, IGrid
import numpy as np
from devito.interfaces import DenseData, TimeData
from TTI_codegen import TTI_cg


dimensions = (50, 50, 50)
model = IGrid()
model.shape = dimensions
origin = (0., 0.)
spacing = (25., 25.)
dtype = np.float32
t_order = 2
spc_order = 2

# True velocity
true_vp = np.ones(dimensions) + 2.0
true_vp[int(dimensions[0] / 3):int(2*dimensions[0]/3), :] = 3.0
true_vp[int(2*dimensions[0] / 3):int(dimensions[0]), :] = 4.0

model.create_model(origin, spacing, true_vp, .3*np.ones(dimensions), .2*np.ones(dimensions), np.pi/3*np.ones(dimensions), np.pi/5*np.ones(dimensions))

# Define seismic data.
data = IShot()

f0 = .010
dt = model.get_critical_dt()
t0 = 0.0
tn = 200.0
nt = int(1+(tn-t0)/dt)
h = model.get_spacing()
data.reinterpolate(dt)
# Set up the source as Ricker wavelet for f0


def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1-2.*r**2)*np.exp(-r**2)
time_series = source(np.linspace(t0, tn, nt), f0)
location = (origin[0] + dimensions[0] * spacing[0] * 0.5, origin[1] + dimensions[1] * spacing[1] * 0.5,
            origin[1] + 2 * spacing[1])
data.set_source(time_series, dt, location)
receiver_coords = np.zeros((101, 3))
receiver_coords[:, 0] = np.linspace(50, 1200, num=101)
receiver_coords[:, 1] = 625
receiver_coords[:, 2] = location[2]
data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 101)

TTI = TTI_cg(model, data, None, None, t_order=2, s_order=4, nbpml=10)
(rec, u, v) = TTI.Forward()

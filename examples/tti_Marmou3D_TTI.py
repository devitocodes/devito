
from containers import IShot, IGrid
import numpy as np
from TTI_codegen import TTI_cg

# Define geometry 
dimensions = (201, 201, 70)
model = IGrid()
model.shape = dimensions
origin = (0., 0., 0.)
spacing = (20., 20., 20.)
t_order = 2
spc_order = 4
f0 = .010
t0 = 0.0
tn = 2000.0

# Read physical parameters

vp = 1e-3*np.fromfile('../../data/marmousi3D/MarmousiVP.raw', dtype='float32', sep="")
vp = vp.reshape(dimensions)
epsilon = np.fromfile('../../data/marmousi3D/MarmousiEps.raw', dtype='float32', sep="")
epsilon = epsilon.reshape(dimensions)
delta = np.fromfile('../../data/marmousi3D/MarmousiDelta.raw', dtype='float32', sep="")
delta = delta.reshape(dimensions)
theta = np.fromfile('../../data/marmousi3D/MarmousiTilt.raw', dtype='float32', sep="")
theta = theta.reshape(dimensions)
phi = np.fromfile('../../data/marmousi3D/Azimuth.raw', dtype='float32', sep="")
phi = phi.reshape(dimensions)

model.create_model(origin, spacing, vp, epsilon, delta, theta, phi)

# Source/receiver geometry and signature
data = IShot()
dt = model.get_critical_dt()
nt = int(1+(tn-t0)/dt)

def source(t, f0):
    r = (np.pi * f0 * (t - 1./f0))
    return (1-2.*r**2)*np.exp(-r**2)
time_series = source(np.linspace(t0, tn, nt), f0)
location = (origin[0] + dimensions[0] * spacing[0] * 0.5, origin[1] + dimensions[1] * spacing[1] * 0.5,
            origin[1] + 2 * spacing[1])
data.set_source(time_series, dt, location)

receiver_coords = np.zeros((201 * 201, 3))
for i in range(0, 201):
    for j in range(0, 201):
        receiver_coords[201 * i + j, 0] = origin[0] + spacing[0] * i
        receiver_coords[201 * i + j, 1] = origin[1] + spacing[1] * j
        receiver_coords[201 * i + j, 2] = location[2]

data.set_receiver_pos(receiver_coords)
data.set_shape(nt, 201 * 201)

# Modelling 
TTI = TTI_cg(model, data, None, None, t_order=2, s_order=6, nbpml=40)
(rec, u, v) = TTI.Forward()

ft = open('RecTTI', 'w')
ft.write(rec.data)
ft.close()
# ft2 = open('Wftti', 'w')
# ft2.write(u.data)
# ft2.close()

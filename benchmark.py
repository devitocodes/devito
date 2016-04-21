from AcousticWave2D_codegen import AcousticWave2D_cg
import numpy as np
from AcousticWave2D import AcousticWave2D
from SubsurfaceModel2D import SubsurfaceModel2D
import time
from SeismicData import SeismicData
from math import floor
from terminaltables import AsciiTable

dimensions = (100, 100)
model = SubsurfaceModel2D(shape=dimensions)
model0 = SubsurfaceModel2D(shape=dimensions)

origin = (0., 0.)
spacing = (25., 25)

# Velocity models
def smooth10(vel, nx, ny):
    out = np.ones((nx, ny))
    out[:, :] = vel[:, :]
    for a in range(5, nx-6):
        out[a, :] = np.sum(vel[a - 5:a + 5, :], axis=0) / 10
    return out

# True velocity
true_vp = np.ones(dimensions) + 2.0
true_vp[floor(dimensions[0] / 2):floor(dimensions[0]), :] = 4.5

# Smooth velocity
initial_vp = smooth10(true_vp, dimensions[0], dimensions[1])
dm = true_vp**-2 - initial_vp**-2
dv = -true_vp + initial_vp
model.create_model(origin, spacing, true_vp)
model0.create_model(origin, spacing, initial_vp)
# Define seismic data.
data = SeismicData()

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
location = (origin[0] + dimensions[0] * spacing[0] * 0.5,
            origin[1] + dimensions[1] * spacing[1] * 0.05)
data.set_source(time_series, dt, location)

data.receiver_coords[0,2] = 10+4
# A Forward propagation example
print "Forward propagation"
print "Starting python lambdified version"
start = time.clock()
(rect, ut) = AcousticWave2D(model, data).Forward(nt)
end = time.clock()
python_time = end-start
norm_rect = np.linalg.norm(rect)
norm_ut = np.linalg.norm(ut)

print "Starting codegen version"

a = AcousticWave2D_cg(model, data)
a.prepare(nt)
start = time.clock()
(recg,ug)= a.Forward()
end = time.clock()
cg_time = end-start
norm_recg = np.linalg.norm(recg)
norm_ug = np.linalg.norm(ug)

table_data = [
    ['', 'Time', 'L2Norm(u)', 'L2Norm(rec)'],
    ['Python lambdified', str(python_time), str(norm_ut), str(norm_rect)],
    ['Codegen', str(cg_time), str(norm_ug), str(norm_recg)]
]
table = AsciiTable(table_data)
print table.table

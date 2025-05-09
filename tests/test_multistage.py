import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

import devito as dv
from examples.seismic import Receiver, TimeAxis

from devito.operator.new_classes import RK, MultiStage
# Set logging level for debugging
dv.configuration['log-level'] = 'DEBUG'

# Parameters
space_order = 2
fd_order = 2
extent = (1000, 1000)
shape = (201, 201)
origin = (0, 0)

# Grid setup
grid = dv.Grid(origin=origin, extent=extent, shape=shape, dtype=np.float64)
x, y = grid.dimensions
dt = grid.stepping_dim.spacing
t = grid.time_dim
dx = extent[0] / (shape[0] - 1)

# Medium velocity model
vel = dv.Function(name="vel", grid=grid, space_order=space_order, dtype=np.float64)
vel.data[:] = 1.0
vel.data[150:, :] = 1.3

# Define wavefield unknowns: u (displacement) and v (velocity)
fun_labels = ['u', 'v']
U = [dv.TimeFunction(name=name, grid=grid, space_order=space_order,
                    time_order=1, dtype=np.float64) for name in fun_labels]

# Time axis
t0, tn = 0.0, 500.0
dt0 = np.max(vel.data) / dx**2
nt = int((tn - t0) / dt0)
dt0 = tn / nt
time_range = TimeAxis(start=t0, stop=tn, num=nt + 1)

# Receiver setup
rec = Receiver(name='rec', grid=grid, npoint=3, time_range=time_range)
rec.coordinates.data[:, 0] = np.linspace(0, 1, 3)
rec.coordinates.data[:, 1] = 0.5
rec = rec.interpolate(expr=U[0].forward)

# Source definition
src_spatial = dv.Function(name="src_spat", grid=grid, space_order=space_order, dtype=np.float64)
src_spatial.data[100, 100] = 1 / dx**2

f0 = 0.01
src_temporal = (1 - 2 * (np.pi * f0 * (t * dt - 1/f0))**2) * sym.exp(-(np.pi * f0 * (t * dt - 1/f0))**2)

# PDE system (2D acoustic)
system_eqs = [U[1],
    (dv.Derivative(U[0], (x, 2), fd_order=fd_order) +
     dv.Derivative(U[0], (y, 2), fd_order=fd_order) +
     src_spatial * src_temporal) * vel**2]

# Time integration scheme
rk = RK.RK44()

# MultiStage object
pdes = [MultiStage(dv.Eq(U[i], system_eqs[i]), rk) for i in range(2)]

# Construct and run operator
op = dv.Operator(pdes + [rec], subs=grid.spacing_map)
op(dt=dt0, time=nt)

# Plot final wavefield
plt.imshow(U[0].data[1, :], cmap="seismic")
plt.colorbar(label="Amplitude")
plt.title("Wavefield snapshot (t = final)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

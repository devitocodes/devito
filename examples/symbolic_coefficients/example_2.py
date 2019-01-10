import numpy as np

from examples.seismic import Model, plot_velocity

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from examples.seismic import RickerSource
from examples.seismic import TimeAxis

from devito import Grid
from devito import Function, TimeFunction
from devito import Eq, solve
from devito import Operator

# Define a physical size
shape = (201, 201)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 2km by 2km
origin = (0., 0.)
extent = (2000, 2000)

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :121] = 1.5
v[:, 121:] = 4.0

grid = Grid(shape=shape, extent=extent)
time = grid.time_dim
x, y = grid.dimensions

# time stepping parameters
t_end = 500.0
dt = 0.2
ns = int(t_end/dt)+1

# Ricker wavelet
f0 = 0.025  # Source peak frequency is 25Hz (0.025 kHz)
src = RickerSource(name='src', grid=grid, f0=f0, npoint=1,
                   time_range=TimeAxis(start=0.0, stop=t_end, step=dt))

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = (1000,1000)
src.coordinates.data[0, -1] = 800.  # Depth is 800m

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=grid, time_order=2, space_order=20, save=ns)
vm = Function(name="vm", grid=grid, space_order=20)

vm.data[:,:] = v[:,:]

# We can now write the PDE
eq = Eq(1.0/vm**2 * u.dt2 - u.laplace)

stencil = solve(eq, u.forward)

# Finally we define the source injection and receiver read function to generate the corresponding code
src_term = src.inject(field=u.forward, expr=src * dt**2 * v**2)

op = Operator([stencil] + src_term)

print("Running model")
op.apply(time_m=0, time_M=ns-1, dt=dt)

##NBVAL_IGNORE_OUTPUT
#from examples.seismic import plot_shotrecord

#plot_shotrecord(rec.data, model, t0, tn)

# u field plot
Lx = 2000
Ly = 2000
#Lx = 2200
#Ly = 2200
dx = 10
dy = dx
X, Y = np.mgrid[0: Lx+1e-10: dx, 0: Ly+1e-10: dy]

#print(u.data[-1,:,:].shape)

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(111)
cont = ax1.contourf(X,Y,u.data[-1,:,:], cmap=cm.binary)
fig.colorbar(cont)
ax1.axis([0, Lx, 0, Ly])
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_title('$u(x,y,t)$')

plt.gca().invert_yaxis()
plt.show()

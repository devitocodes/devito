import numpy as np
#%matplotlib inline

#NBVAL_IGNORE_OUTPUT
from examples.seismic import Model, plot_velocity

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

# Define a physical size
shape = (201, 201)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 2km by 2km
origin = (0., 0.)  # What is the location of the top left corner. This is necessary to define
# the absolute location of the source and receivers

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :121] = 1.5
v[:, 121:] = 4.0

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as 10 grid points
#model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              #space_order=20, nbpml=10)
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=20, nbpml=0)

#plot_velocity(model)

from examples.seismic import TimeAxis

t0 = 0.0  # Simulation starts a t=0
tn = 500.  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing
#dt = 0.1

time_range = TimeAxis(start=t0, stop=tn, step=dt)

#NBVAL_IGNORE_OUTPUT
from examples.seismic import RickerSource

f0 = 0.025  # Source peak frequency is 25Hz (0.025 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[0, -1] = 800.  # Depth is 800m

# We can plot the time signature to see the wavelet
#src.show()

#NBVAL_IGNORE_OUTPUT
from examples.seismic import Receiver

# Create symbol for 101 receivers
#rec = Receiver(name='rec', grid=model.grid, npoint=101, time_range=time_range)

# Prescribe even spacing for receivers along the x-axis
#rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
#rec.coordinates.data[:, 1] = 20.  # Depth is 20m

# We can now show the source and receivers within our domain:
# Red dot: Source location
# Green dots: Receiver locations (every 4th point)
#plot_velocity(model, source=src.coordinates.data,
              #receiver=rec.coordinates.data[::4, :])

# In order to represent the wavefield u and the square slowness we need symbolic objects 
# corresponding to time-space-varying field (u, TimeFunction) and 
# space-varying field (m, Function)
from devito import TimeFunction

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=20)

# We can now write the PDE
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

# This discrete PDE can be solved in a time-marching way updating u(t+dt) from the previous time step
# Devito as a shortcut for u(t+dt) which is u.forward. We can then rewrite the PDE as 
# a time marching updating equation known as a stencil using customized SymPy functions
from devito import Eq, solve

stencil = Eq(u.forward, solve(pde, u.forward))

# Finally we define the source injection and receiver read function to generate the corresponding code
src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)

# Create interpolation expression for receivers
#rec_term = rec.interpolate(expr=u.forward)

#NBVAL_IGNORE_OUTPUT
from devito import Operator

#op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)
op = Operator([stencil] + src_term, subs=model.spacing_map)

#NBVAL_IGNORE_OUTPUT
#op(time=time_range.num-1, dt=model.critical_dt)
op(time=time_range.num-1, dt=dt)

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

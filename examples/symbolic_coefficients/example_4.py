import numpy as np
#%matplotlib inline

#NBVAL_IGNORE_OUTPUT
from examples.seismic import Model, plot_velocity

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import sys

# Define a physical size
Lx = 2000
Ly = Lx
h = 1
Nx = int(Lx/h)+1
Ny = Nx

shape = (Nx, Ny)  # Number of grid point (nx, nz)
spacing = (h, h)  # Grid spacing in m. The domain size is now 2km by 2km
origin = (0., 0.)

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :int(0.6*(Nx-1)+1)] = 1.5
v[:, int(0.6*(Nx-1)+1):] = 4.0

nbpml = 20
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=20, nbpml=nbpml)

#plot_velocity(model)

from examples.seismic import TimeAxis

t0 = 0.0  # Simulation starts a t=0
tn = 500.  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing
#dt = 0.2

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

from devito import TimeFunction

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=20)

m = model.m

#print(m.data[:])
#sys.exit()

# We can now write the PDE
#pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
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

abs_lay = nbpml*h

dx = h
dy = dx
X, Y = np.mgrid[-abs_lay: Lx+abs_lay+1e-10: dx, -abs_lay: Ly+abs_lay+1e-10: dy]

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

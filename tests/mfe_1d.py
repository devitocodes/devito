import numpy as np

from devito import (SpaceDimension, Grid, TimeFunction, Eq, Operator,
                    solve, Constant, norm)
from examples.seismic.source import TimeAxis, Receiver

# Space related
extent = (1500., )
shape = (201, )
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, ))

# Time related
t0, tn = 0., 30.
dt = (10. / np.sqrt(2.)) / 6.
time_range = TimeAxis(start=t0, stop=tn, step=dt)

# Velocity and pressure fields
so = 2
to = 1
v = TimeFunction(name='v', grid=grid, space_order=so, time_order=to)
tau = TimeFunction(name='tau', grid=grid, space_order=so, time_order=to)

# The receiver
nrec = 1
rec = Receiver(name="rec", grid=grid, npoint=nrec, time_range=time_range)
rec.coordinates.data[:, 0] = np.linspace(0., extent[0], num=nrec)
rec_term = rec.interpolate(expr=v)

# First order elastic-like dependencies equations
pde_v = v.dt - (tau.dx)
pde_tau = (tau.dt - ((v.forward).dx))

u_v = Eq(v.forward, solve(pde_v, v.forward))
u_tau = Eq(tau.forward, solve(pde_tau, tau.forward))

op = Operator([u_v] + [u_tau] + rec_term)
op.apply(dt=dt)

print(norm(v))
print(norm(tau))
# print(op.ccode)
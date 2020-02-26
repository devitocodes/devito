from examples.cfd import plot_field, init_hat
import numpy as np


from devito import Grid, TimeFunction, Eq, solve, Operator

# Some variable declarations
nx = 2180
ny = 2180
nt = 100
c = 1.
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
print("dx %s, dy %s" % (dx, dy))
sigma = .2
dt = sigma * dx

grid = Grid(shape=(nx, ny), extent=(2., 2.))
u = TimeFunction(name='u', grid=grid)
init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
eq = Eq(u.dt + c*u.dxl + c*u.dyl, subdomain=grid.interior)

print(eq)
stencil = solve(eq, u.forward)

# Reset our initial condition in both buffers.
# This is required to avoid 0s propagating into
# our solution, which has a background value of 1.
init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
init_hat(field=u.data[1], dx=dx, dy=dy, value=2.)

# Create an operator that updates the forward stencil point
op = Operator(Eq(u.forward, stencil, subdomain=grid.interior))

# Apply the operator for a number of timesteps
op(time=nt, dt=dt)

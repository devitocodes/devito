# A 2D heat diffusion using Devito
# BC modelling included
# PyVista plotting included

import argparse
import numpy as np

from devito import Grid, TimeFunction, Eq, solve, Operator, Constant, norm, XDSLOperator
from examples.seismic import plot_image
from examples.cfd import init_hat

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=1,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=40,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot3D")
args = parser.parse_args()

# Some variable declarations
nx, ny = args.shape
nt = args.nt
nu = .5
dx = 1. / (nx - 1)
dy = 1. / (ny - 1)
sigma = .25

dt = sigma * dx * dy / nu
so = args.space_order
to = args.time_order

print("dx %s, dy %s" % (dx, dy))

grid = Grid(shape=(nx, ny), extent=(2., 2.))
u = TimeFunction(name='u', grid=grid, space_order=so)

# Reset our data field and ICs
# u.data[:, :, :] = 0.1
init_hat(field=u.data[0], dx=dx, dy=dy, value=1.)


# u.data[0, :, int(ny/2)] = 2

a = Constant(name='a')
# Create an equation with second-order derivatives
eq = Eq(u.dt, a * u.laplace, subdomain=grid.interior)
stencil = solve(eq, u.forward)
eq_stencil = Eq(u.forward, stencil)

# Create boundary condition expressions
x, y = grid.dimensions
t = grid.stepping_dim

# Add boundary conditions
# bc = [Eq(u[t+1, x, y, 0], 2.)]  # bottom
# bc += [Eq(u[t+1, x, y, nz-1], 2.)]  # top
# bc += [Eq(u[t+1, 0, y, z], 2.)]  # left
# bc += [Eq(u[t+1, nx-1, y, z], 2.)]  # right

# bc += [Eq(u[t+1, x, 0, z], 2.)]  # front
# bc += [Eq(u[t+1, x, ny-1, z], 2.)]  # back

print(eq_stencil)

# Create an operator that updates the forward stencil point
# plus adding boundary conditions
# op = Operator([eq_stencil] + bc, subdomain=grid.interior)

# No BCs
# op = XDSLOperator([eq_stencil])
op = Operator([eq_stencil])
# print(op.ccode)
# dt = 0.00002
# import pdb;pdb.set_trace()
# Apply the operator for a number of timesteps
print(dt)
op.apply(time=nt, dt=dt, a=nu)

print("Field norm is:", norm(u))

plot_image(u.data[0], cmap="seismic")

# import pdb;pdb.set_trace()

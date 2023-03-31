# A 3D heat diffusion using Devito
# BC modelling included
# PyVista plotting included

import argparse
import numpy as np

from devito import (Grid, Eq, TimeFunction, Operator,
                    Constant, solve, XDSLOperator)
from devito.logger import info, perf
from devito.ir.ietxdsl.cluster_to_ssa import generate_launcher_base

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(110, 110, 110), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=4,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=1,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=40,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot3D")

parser.add_argument("-xdsl", "--xdsl", default=False, action='store_true')
args = parser.parse_args()

# Some variable declarations
nx, ny, nz = args.shape
nt = args.nt
nu = .5
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
dz = 2. / (nz - 1)
sigma = .25
dt = sigma * dx * dz * dy / nu
so = args.space_order
to = 1
print("dx %s, dy %s, dz %s" % (dx, dy, dz))

grid = Grid(shape=(nx, ny, nz), extent=(2., 2., 2.))
u = TimeFunction(name='u', grid=grid, space_order=so)
# init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
u.data[:, :, :, :] = 0
u.data[:, int(nx/2), :, :] = 1

u.data_with_halo[0, :, :, :].tofile('input.data')

a = Constant(name='a')
# Create an equation with second-order derivatives
eq = Eq(u.dt, a * u.laplace, subdomain=grid.interior)
stencil = solve(eq, u.forward)
eq_stencil = Eq(u.forward, stencil)

# Create boundary condition expressions
x, y, z = grid.dimensions
t = grid.stepping_dim

# Add boundary conditions
# bc = [Eq(u[t+1, x, y, 0], 2.)]  # bottom
# bc += [Eq(u[t+1, x, y, nz-1], 2.)]  # top
# bc += [Eq(u[t+1, 0, y, z], 2.)]  # left
# bc += [Eq(u[t+1, nx-1, y, z], 2.)]  # right

# bc += [Eq(u[t+1, x, 0, z], 2.)]  # front
# bc += [Eq(u[t+1, x, ny-1, z], 2.)]  # back

# Create an operator that updates the forward stencil point
# plus adding boundary conditions
# op = Operator([eq_stencil] + bc, subdomain=grid.interior)
if args.xdsl:
    perf("Generating XDSLOperator")
    xop = XDSLOperator([eq_stencil], subdomain=grid.interior)
    info("Operator in main.mlir")
    with open("main.mlir", "w") as f:
        f.write(generate_launcher_base(xop._module, {
            'time_m': 0,
            'time_M': nt,
            **{str(k): float(v) for k, v in dict(grid.spacing_map).items()},
            'a': nu,
            'dt': dt,
        }, u.shape_allocated[1:]))

    info("Operator in kernel.mlir")
    with open("kernel.mlir", "w") as f:
        f.write(xop.mlircode)


    import sys;sys.exit(0)

# No BCs
op = Operator([eq_stencil], subdomain=grid.interior)

# Apply the operator for a number of timesteps
op(time=nt, dt=dt, a=nu)

# get final data step
# this is cursed math, but we assume that:
#  1. Every kernel always writes to t1
#  2. The formula for calculating t1 = (time + n - 1) % n, where n is the number of time steps we have
#  3. the loop goes for (...; time <= time_M; ...), which means that the last value of time is time_M
#  4. time_M is always nt in this example
t1 = (nt + u._time_size - 1)%(2)

res_data: np.array = u.data[t1,:,:,:]

datafile = 'devito.data'
info("Save result data to " + datafile)
res_data.tofile(datafile)

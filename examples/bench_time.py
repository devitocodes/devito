import pytest
import numpy as np

from devito import (Grid, Eq, Function, TimeFunction, Operator, norm,
                    Constant, solve)
from devito.ir import Expression, Iteration, FindNodes, FindSymbols
import argparse

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=4,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=1,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=40,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
args = parser.parse_args()


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

# Initialise u
init_value = 6.5

# Field initialization
grid = Grid(shape=(nx, ny, nz))
u = TimeFunction(name='u', grid=grid, space_order=so, time_order=to)
u.data[:, :, :] = init_value

# Create an equation with second-order derivatives
a = Constant(name='a')
eq = Eq(u.dt, a*u.laplace + 0.1, subdomain=grid.interior)
stencil = solve(eq, u.forward)
eq0 = Eq(u.forward, stencil)

# List comprehension would need explicit locals/globals mappings to eval
op0 = Operator(eq0, opt=('advanced'))
op0.apply(time_M=nt, dt=dt)
norm_u = norm(u)
u.data[:] = init_value

op1 = Operator(eq0, opt=('advanced', {'skewing': True,
                         'blocklevels': 2}))

op1.apply(time_M=nt, dt=dt)
print(norm_u)
print(norm(u))
assert np.isclose(norm(u), norm_u, atol=1e-4, rtol=0)

iters = FindNodes(Iteration).visit(op1)

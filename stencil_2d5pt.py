import numpy as np

from devito import (Grid, Eq, TimeFunction, Operator, norm,
                    Constant, solve, XDSLOperator)
from devito.ir import Iteration, FindNodes
import argparse

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=1,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=10,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
args = parser.parse_args()


nx, ny = args.shape
nt = args.nt
nu = .5
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu

so = args.space_order
to = 2

# Initialise u
init_value = 10

# Field initialization
grid = Grid(shape=(nx, ny))
u = TimeFunction(name='u', grid=grid, space_order=so, time_order=to)
u.data[:, :, :] = 0
u.data[:, int(nx/2), int(nx/2)] = init_value
u.data[:, int(nx/2), -int(nx/2)] = -init_value

# Create an equation with second-order derivatives
a = Constant(name='a')
eq = Eq(u.dt, a*u.laplace + 0.01)
stencil = solve(eq, u.forward)
eq0 = Eq(u.forward, stencil)

xop = XDSLOperator([eq0])
print(xop.ccode)
# xop.apply(time_M=nt, a=0.1, dt=dt)
xop.apply(time_M=nt, a=0.1, dt=dt)
xdsl_data: np.array = u.data_with_halo.copy()
xdsl_norm = norm(u)


u.data[:, :, :] = 0
u.data[:, 10, 10] = init_value
u.data[:, 10, -10] = -init_value
print(f"Original norm is {norm(u)}")


op = Operator([eq0])
op.apply(time_M=nt, a=0.1, dt=dt)
orig_data: np.array = u.data_with_halo.copy()
orig_norm = norm(u)

print("orig={}, xdsl={}".format(xdsl_norm, orig_norm))
assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()


from examples.cfd import plot_field

print("After", nt, "timesteps")
plot_field(u.data[0], zmax=4.5)

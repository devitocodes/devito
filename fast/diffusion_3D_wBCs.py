# A 3D heat diffusion using Devito
# BC modelling included
# PyVista plotting included

import argparse
import numpy as np

from devito import (Grid, TimeFunction, Eq, solve, Constant,
                    norm, Operator, configuration)
from fast.bench_utils import plot_3dfunc

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
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot3D")
parser.add_argument("-devito", "--devito", default=False, type=bool, help="Devito run")
parser.add_argument("-xdsl", "--xdsl", default=False, type=bool, help="xDSL run")
args = parser.parse_args()

mpiconf = configuration['mpi']

# Some variable declarations
nx, ny, nz = args.shape
nt = args.nt
nu = .9
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
dz = 2. / (nz - 1)

sigma = .25

dt = sigma * dx * dz * dy / nu
so = args.space_order
to = args.time_order

# print("dx %s, dy %s, dz %s" % (dx, dy, dz))

grid = Grid(shape=(nx, ny, nz), extent=(2., 2., 2.))
u = TimeFunction(name='u', grid=grid, space_order=so)
devito_out = TimeFunction(name='u', grid=grid, space_order=so)

# Create an equation with second-order derivatives
a = Constant(name='a')
eq = Eq(u.dt, a * u.laplace)
stencil = solve(eq, u.forward)
eq_stencil = Eq(u.forward, stencil)


# Create Operator
if args.devito:
    u.data[:, :, :, :] = 0
    u.data[:, :, :, int(nz/2)] = 1

    # To measure Devito at its best on GPU, we have to set the tile siwe manually
    opt = 'advanced'
    if configuration['platform'].name == 'nvidiaX':
        opt = ('advanced', {'par-tile': (32, 4, 8)})
    op = Operator([eq_stencil], name='DevitoDiffusionOperator', opt=opt)

    # Apply the operator for a number of timesteps
    op.apply(time=nt, dt=dt, a=nu)
    print("Devito Field norm is:", norm(u))
    if args.plot:
        plot_3dfunc(u)

    if args.xdsl:
        configuration['mpi'] = 0
        devito_out.data[:] = u.data[:]
        configuration['mpi'] = mpiconf

if args.xdsl:
    # Reset field
    u.data[:, :, :, :] = 0
    u.data[:, :, :, int(nz/2)] = 1
    op = Operator([eq_stencil], name='xDSLDiffusionOperator', opt='xdsl')
    # Apply the xdsl operator for a number of timesteps
    op.apply(time=nt, dt=dt, a=nu)
    print("XDSL Field norm is:", norm(u))
    if args.plot:
        plot_3dfunc(u)

if args.xdsl and args.devito:
    assert np.isclose(norm(u), norm(devito_out), rtol=1e-5)
    max_error = np.max(np.abs(u.data - devito_out.data))
    print("Max error: ", max_error)

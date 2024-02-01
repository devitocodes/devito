# A 2D heat diffusion using Devito
# BC modelling included
# PyVista plotting included

import argparse
import numpy as np

from devito import (Grid, TimeFunction, Eq, solve, Operator,
                    Constant, norm, configuration)
from examples.cfd import init_hat
from fast.bench_utils import plot_2dfunc

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
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot2D")
parser.add_argument("-devito", "--devito", default=False, type=bool, help="Devito run")
parser.add_argument("-xdsl", "--xdsl", default=False, type=bool, help="xDSL run")
args = parser.parse_args()

mpiconf = configuration['mpi']

# flake8: noqa

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
u2 = TimeFunction(name='u', grid=grid, space_order=so)
devito_out = TimeFunction(name='u', grid=grid, space_order=so)

a = Constant(name='a')
# Create an equation with second-order derivatives
# eq = Eq(u.dt, a * u.laplace, subdomain=grid.interior)
eq = Eq(u.dt, a * u.laplace)
stencil = solve(eq, u.forward)
eq_stencil = Eq(u.forward, stencil)

# Reset our data field and ICs
init_hat(field=u.data[0], dx=dx, dy=dy, value=1.)

if args.xdsl and args.devito:
    configuration['mpi'] = 0
    u2.data[:] = u.data[:]
    configuration['mpi'] = mpiconf

if args.devito:

    # To measure Devito at its best on GPU, we have to set the tile siwe manually
    opt = 'advanced'
    if configuration['platform'].name == 'nvidiaX':
        opt = ('advanced', {'par-tile': (32, 4, 8)})

    op = Operator([eq_stencil], name='DevitoDiffusionOperator', opt=opt)
    op.apply(time=nt, dt=dt, a=nu)
    print("Devito Field norm is:", norm(u))

    if args.plot:
        plot_2dfunc(u)

    if args.xdsl:
        configuration['mpi'] = 0
        devito_out.data[:] = u.data[:]
        u.data[:] = u2.data[:]
        configuration['mpi'] = mpiconf

# Reset our data field and ICs
init_hat(field=u.data[0], dx=dx, dy=dy, value=1.)

if args.xdsl:
    op = Operator([eq_stencil], name='XDSLDiffusionOperator', opt='xdsl')
    op.apply(time=nt, dt=dt, a=nu)
    print("XDSL Field norm is:", norm(u))

if args.xdsl and args.devito:
    assert np.isclose(norm(u), norm(devito_out), rtol=1e-5)
    max_error = np.max(np.abs(u.data - devito_out.data))
    print("Max error: ", max_error)

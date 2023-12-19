# Based on the implementation of the Devito acoustic example implementation
# Not using Devito's source injection abstraction
import numpy as np

from devito import (TimeFunction, Eq, Operator, solve, norm,
                    configuration, Grid)
from devito.tools import as_tuple

import argparse
from fast.bench_utils import plot_2dfunc

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(16, 16), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=4,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=2,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=20,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=1, type=int, nargs="+",
                    help="Block levels")
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot2D")
parser.add_argument("-devito", "--devito", default=False, type=bool, help="Devito run")
parser.add_argument("-xdsl", "--xdsl", default=False, type=bool, help="xDSL run")
args = parser.parse_args()

mpiconf = configuration['mpi']

# Define a physical size
# nx, ny, nz = args.shape
nt = args.nt
so = args.space_order

shape = (args.shape)  # Number of grid point (nx, ny, nz)
shape_str = '_'.join(str(item) for item in shape)
# Grid spacing in m. The domain size is now 1km by 1km
spacing = as_tuple(10.0 for _ in range(len(shape)))
# What is the location of the top left corner.
origin = as_tuple(0.0 for _ in range(len(shape)))
domain_size = tuple((d-1) * s for d, s in zip(shape, spacing))
extent = np.load("so%s_grid_extent%s.npy" % (so, shape_str), allow_pickle=True)

grid = Grid(shape=shape, extent=as_tuple(extent))

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as
# 10 grid points
so = args.space_order
to = args.time_order

t0 = 0.  # Simulation starts a t=0
tn = nt  # Simulation last 1 second (1000 ms)

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=grid, time_order=to, space_order=so)
# Another one to clone data
u2 = TimeFunction(name="u", grid=grid, time_order=to, space_order=so)
devito_out = TimeFunction(name="u", grid=grid, time_order=to, space_order=so)

# We can now write the PDE
# pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
pde = u.dt2 - u.laplace

stencil = Eq(u.forward, solve(pde, u.forward))

# print("Init Devito linalg norm 0 :", np.linalg.norm(u.data[0]))
# print("Init Devito linalg norm 1 :", np.linalg.norm(u.data[1]))
# print("Init Devito linalg norm 2 :", np.linalg.norm(u.data[2]))
# print("Norm of initial data:", norm(u))

u.data[:] = np.load("so%s_wave_dat%s.npy" % (so, shape_str), allow_pickle=True)
dt = np.load("so%s_critical_dt%s.npy" % (so, shape_str), allow_pickle=True)

if args.xdsl and args.devito:
    configuration['mpi'] = 0
    u2.data[:] = u.data[:]
    configuration['mpi'] = mpiconf

# np.save("critical_dt%s.npy" % shape_str, model.critical_dt, allow_pickle=True)
# np.save("wave_dat%s.npy" % shape_str, u.data[:], allow_pickle=True)

if len(shape) == 2 and args.plot:
    plot_2dfunc(u)

print("Init norm:", np.linalg.norm(u.data[:]))

if args.devito:
    # To measure Devito at its best on GPU, we have to set the tile siwe manually
    opt = 'advanced'
    if configuration['platform'].name == 'nvidiaX':
        opt = ('advanced', {'par-tile': (32, 4, 8)})

    # Run more with no sources now (Not supported in xdsl)
    # op1 = Operator([stencil], name='DevitoOperator', subs=grid.spacing_map)
    op1 = Operator([stencil], name='DevitoOperator', opt=opt)
    op1.apply(time=nt, dt=dt)

    if len(shape) == 2 and args.plot:
        plot_2dfunc(u)

    print("Devito norm:", norm(u))

    if args.xdsl:
        configuration['mpi'] = 0
        devito_out.data[:] = u.data[:]
        u.data[:] = u2.data[:]
        configuration['mpi'] = mpiconf

    # print("Devito linalg norm 0:", np.linalg.norm(u.data[0]))
    # print("Devito linalg norm 1:", np.linalg.norm(u.data[1]))
    # print("Devito linalg norm 2:", np.linalg.norm(u.data[2]))


if args.xdsl:
    # print("Reinitialise data: Devito norm:", norm(u))
    # print("XDSL init linalg norm:", np.linalg.norm(u.data[0]))
    # print("XDSL init linalg norm:", np.linalg.norm(u.data[1]))
    # print("XDSL init linalg norm:", np.linalg.norm(u.data[2]))

    # Run more with no sources now (Not supported in xdsl)
    xdslop = Operator([stencil], name='xDSLOperator', opt='xdsl')
    xdslop.apply(time=nt, dt=dt)

    if len(shape) == 2 and args.plot:
        plot_2dfunc(u)

    print("XDSL norm:", norm(u))

    # print("XDSL output norm 0:",
    #       np.linalg.norm(u.data[0]),
    #       "vs:",
    #       np.linalg.norm(ub.data[0]))
    # print("XDSL output norm 1:",
    #       np.linalg.norm(u.data[1]),
    #       "vs:",
    #       np.linalg.norm(ub.data[1]))
    # print("XDSL output norm 2:",
    #       np.linalg.norm(u.data[2]),
    #       "vs:",
    #       np.linalg.norm(ub.data[2]))

if args.xdsl and args.devito:
    max_error = np.max(np.abs(u.data - devito_out.data))
    print("Max error: ", max_error)
    assert np.isclose(norm(u), norm(devito_out), rtol=1e-2)

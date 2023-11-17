# Script to save initial data for the Acoustic wave execution benchmark
# Based on the implementation of the Devito acoustic example implementation
# Not using Devito's source injection abstraction
import numpy as np

from devito import (TimeFunction, Eq, Operator, solve, configuration)
from examples.seismic import RickerSource
from examples.seismic import Model, TimeAxis
from fast.bench_utils import plot_2dfunc
from devito.tools import as_tuple

import argparse
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

shape = (args.shape)  # Number of grid point (nx, ny, nz)
# Grid spacing in m. The domain size is now 1km by 1km
spacing = as_tuple(10.0 for _ in range(len(shape)))
# What is the location of the top left corner.
origin = as_tuple(0.0 for _ in range(len(shape)))
# This is necessary to define the absolute location of the
# source and receivers

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :] = 1

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as
# 10 grid points
so = args.space_order
to = args.time_order

model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=so, nbl=0)

# plot_velocity(model)

t0 = 0.  # Simulation starts a t=0
tn = nt  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing
print("dt is:", dt)

time_range = TimeAxis(start=t0, stop=tn, step=dt)

# The source is positioned at a $20m$ depth and at the middle of the
# $x$ axis ($x_{src}=500m$),
# with a peak wavelet frequency of $10Hz$.
f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .5

# We can plot the time signature to see the wavelet
# src.show()

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=to, space_order=so)
# Another one to clone data
u2 = TimeFunction(name="u", grid=model.grid, time_order=to, space_order=so)
ub = TimeFunction(name="ub", grid=model.grid, time_order=to, space_order=so)

# We can now write the PDE
# pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
pde = u.dt2 - u.laplace

stencil = Eq(u.forward, solve(pde, u.forward))
# stencil

# Finally we define the source injection and receiver read function to generate
# the corresponding code
# print(time_range)

print("Init norm:", np.linalg.norm(u.data[:]))
src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
op0 = Operator([stencil] + src_term, subs=model.spacing_map, name='SourceDevitoOperator')

# Run with source and plot
op0.apply(time=time_range.num-1, dt=model.critical_dt)

if len(shape) == 2:
    if args.plot:
        plot_2dfunc(u)

# Save Data here
shape_str = '_'.join(str(item) for item in shape)
np.save("so%s_critical_dt%s.npy" % (so, shape_str), model.critical_dt, allow_pickle=True)
np.save("so%s_wave_dat%s.npy" % (so, shape_str), u.data[:], allow_pickle=True)
np.save("so%s_grid_extent%s.npy" % (so, shape_str), model.grid.extent, allow_pickle=True)

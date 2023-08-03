# Based on the implementation of the Devito acoustic example implementation
# Not using Devito's source injection abstraction
import sys
import numpy as np
from devito import (TimeFunction, Eq, Operator, solve, norm,
                    XDSLOperator, configuration)
from examples.seismic import RickerSource
from examples.seismic import Model, TimeAxis

from devito.tools import as_tuple

import argparse
np.set_printoptions(threshold=np.inf)


parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=4,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=2,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=200,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot3D")
args = parser.parse_args()


def plot_3dfunc(u):
    # Plot a 3D structured grid using pyvista

    import matplotlib.pyplot as plt
    import pyvista as pv
    cmap = plt.colormaps["viridis"]
    values = u.data[0, :, :, :]
    vistagrid = pv.ImageData()
    vistagrid.dimensions = np.array(values.shape) + 1
    vistagrid.spacing = (1, 1, 1)
    vistagrid.origin = (0, 0, 0)  # The bottom left corner of the data set
    vistagrid.cell_data["values"] = values.flatten(order="F")
    vistaslices = vistagrid.slice_orthogonal()
    # vistagrid.plot(show_edges=True)
    vistaslices.plot(cmap=cmap)


# Define a physical size
# nx, ny, nz = args.shape
nt = args.nt

shape = (args.shape)  # Number of grid point (nx, ny, nz)
spacing = as_tuple(10.0 for _ in range(len(shape)))  # Grid spacing in m. The domain size is now 1km by 1km
origin = as_tuple(0.0 for _ in range(len(shape)))  # What is the location of the top left corner.
# This is necessary to define
# the absolute location of the source and receivers

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :, :] = 1

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

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=to, space_order=so)
# Another one to clone data
u2 = TimeFunction(name="u", grid=model.grid, time_order=to, space_order=so)
ub = TimeFunction(name="ub", grid=model.grid, time_order=to, space_order=so)


# We can now write the PDE
# pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
# import pdb;pdb.set_trace()
pde = u.dt2 - u.laplace

stencil = Eq(u.forward, solve(pde, u.forward))

src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
op0 = Operator([stencil] + src_term, subs=model.spacing_map, name='SourceDevitoOperator')
# Run with source and plot
op0.apply(time=time_range.num-1, dt=model.critical_dt)


if len(shape) == 3:
    if args.plot:
        plot_3dfunc(u)

# devito_norm = norm(u)
# print("Init linalg norm 0 (inlined) :", norm(u))
# print("Init linalg norm 0 :", np.linalg.norm(u.data[0]))
# print("Init linalg norm 1 :", np.linalg.norm(u.data[1]))
# print("Init linalg norm 2 :", np.linalg.norm(u.data[2]))
# print("Norm of initial data:", np.linalg.norm(u.data[:]))

configuration['mpi'] = 0
u2.data[:] = u.data[:]
configuration['mpi'] = 'basic'

# Run more with no sources now (Not supported in xdsl)
op1 = Operator([stencil], name='DevitoOperator')
op1.apply(time=time_range.num-1, dt=model.critical_dt)

configuration['mpi'] = 0
ub.data[:] = u.data[:]
configuration['mpi'] = 'basic'

if len(shape) == 3:
    if args.plot:
        plot_3dfunc(u)

# print("After Operator 1: Devito norm:", np.linalg.norm(u.data[:]))
#print("Devito norm 0:", np.linalg.norm(u.data[0]))
#print("Devito norm 1:", np.linalg.norm(u.data[1]))
#print("Devito norm 2:", np.linalg.norm(u.data[2]))

# Reset initial data
configuration['mpi'] = 0
u.data[:] = u2.data[:]
configuration['mpi'] = 'basic'

# print("Reinitialise data for XDSL:", np.linalg.norm(u.data[:]))
# print("Init XDSL linalg norm 0:", np.linalg.norm(u.data[0]))
# print("Init XDSL linalg norm 1:", np.linalg.norm(u.data[1]))
# print("Init XDSL linalg norm 2:", np.linalg.norm(u.data[2]))

# Run more with no sources now (Not supported in xdsl)
xdslop = XDSLOperator([stencil], name='XDSLOperator')
xdslop.apply(time=time_range.num-1, dt=model.critical_dt)

print("XDSL output norm 0:", np.linalg.norm(u.data[0]), "vs:", np.linalg.norm(ub.data[0]))
print("XDSL output norm 1:", np.linalg.norm(u.data[1]), "vs:", np.linalg.norm(ub.data[1]))
print("XDSL output norm 2:", np.linalg.norm(u.data[2]), "vs:", np.linalg.norm(ub.data[2]))

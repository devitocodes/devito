# Based on the implementation of the Devito acoustic example implementation
# Not using Devito's source injection abstraction
import numpy as np
from devito import TimeFunction, Eq, Operator, solve, norm, XDSLOperator
from examples.seismic import RickerSource
from examples.seismic import Model, TimeAxis

from devito.tools import as_tuple

import argparse

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
    vistagrid = pv.UniformGrid()
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
v[:, ..., :] = 1

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
#src.show()

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=to, space_order=so)

# We can now write the PDE
# pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
# import pdb;pdb.set_trace()
pde = u.dt2 - u.laplace

# The PDE representation is as on paper
pde

stencil = Eq(u.forward, solve(pde, u.forward))
stencil

# Finally we define the source injection and receiver read function to generate
# the corresponding code
print(time_range)
src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
op = Operator([stencil] + src_term, subs=model.spacing_map, name='DevitoOperator')
# Run with source and plot
op.apply(time=time_range.num-1, dt=model.critical_dt)

if len(shape) == 3:
    if args.plot:
        plot_3dfunc(u)

initdata = u.data[:]

# Run more with no sources now (Not supported in xdsl)
op = Operator([stencil], name='DevitoOperator', opt='noop')
op.apply(time=time_range.num-1, dt=model.critical_dt)

if len(shape) == 3:
    if args.plot:
        plot_3dfunc(u)


devito_output = u.copy()
print("Devito norm:", norm(u))
print(f"devito output norm: {norm(devito_output)}")

# Reset initial data
u.data[:] = initdata

# Run more with no sources now (Not supported in xdsl)
xdslop = XDSLOperator([stencil], name='xDSLOperator')
xdslop.apply(time=time_range.num-1, dt=model.critical_dt)

xdsl_output = u.copy()
print("XDSL norm:", norm(u))
print(f"xdsl output norm: {norm(xdsl_output)}")

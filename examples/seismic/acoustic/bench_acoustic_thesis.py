# Based on the implementation of the Devito acoustic example implementation
import numpy as np
from devito import TimeFunction, Eq, Operator, solve, norm
from examples.seismic import Model, TimeAxis

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=4,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=2,
                    type=int, help="Time order of the simulation")
parser.add_argument("-tn", "--tn", default=40,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
args = parser.parse_args()


# Define a physical size
nx, ny, nz = args.shape
nt = args.tn

shape = (nx, ny, nz)  # Number of grid point (nx, ny, nz)
spacing = (10., 10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0., 0.)  # What is the location of the top left corner. This is necessary to define
# the absolute location of the source and receivers

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :, :51] = 1.5
v[:, :, 51:] = 2.5

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as 10 grid points
so = args.space_order
to = args.time_order

model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=so, nbl=10, bcs="damp")

# plot_velocity(model)

t0 = 0.  # Simulation starts a t=0
tn = nt  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing

time_range = TimeAxis(start=t0, stop=tn, step=dt)

# Define the wavefield with the size of the model and the time dimension
u = TimeFunction(name="u", grid=model.grid, time_order=to, space_order=so)

u.data[0, int(nx/2), int(ny/3), 10] = 5
u.data[0, int(nx/3), int(ny/2), 10] = 1

# We can now write the PDE
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

# The PDE representation is as on paper
pde

stencil = Eq(u.forward, solve(pde, u.forward))
stencil

op = Operator([stencil], subs=model.spacing_map)
op.apply(time=time_range.num-1, dt=model.critical_dt)
# op.apply(time=time_range.num-1, dt=model.critical_dt, **{'x0_blk0_size': 16, 'y0_blk0_size': 8})
print(norm(u))

# Initialize data
u.data[:] = 0
u.data[0, int(nx/2), int(ny/3), 10] = 5
u.data[0, int(nx/3), int(ny/2), 10] = 1

op1 = Operator([stencil], subs=model.spacing_map,
               opt=('advanced', {'skewing': True, 'blocklevels': 2}))
# op1.apply(time=time_range.num-2, dt=model.critical_dt, **{'time0_blk0_size': 282, 'x0_blk0_size': 32, 'x0_blk1_size': 4, 'y0_blk0_size': 32, 'y0_blk1_size': 4})
op1.apply(time=time_range.num-2, dt=model.critical_dt)
print(norm(u))

# import pdb;pdb.set_trace()

import matplotlib.pyplot as plt
from examples.cfd import plot_field
from examples.seismic import plot_image

# r_square = (u.data[0])
# plot_3D_array_slices(r_square)

plot_image(u.data[0,:,:,10], cmap="viridis")
plt.show()

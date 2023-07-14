# A 3D heat diffusion using Devito
# BC modelling included
# PyVista plotting included

import argparse
import numpy as np

from devito import Grid, TimeFunction, Eq, solve, Operator, Constant, norm, XDSLOperator

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
args = parser.parse_args()


def plot_3dfunc(u):
    # Plot a 3D structured grid using pyvista

    import matplotlib.pyplot as plt
    import pyvista as pv

    cmap = plt.cm.get_cmap("viridis")
    values = u.data[0, :, :, :]
    vistagrid = pv.ImageData()
    vistagrid.dimensions = np.array(values.shape) + 1
    vistagrid.spacing = (1, 1, 1)
    vistagrid.origin = (0, 0, 0)  # The bottom left corner of the data set
    vistagrid.cell_data["values"] = values.flatten(order="F")
    vistaslices = vistagrid.slice_orthogonal()
    # vistagrid.plot(show_edges=True)
    vistaslices.plot(cmap=cmap)


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

print("dx %s, dy %s, dz %s" % (dx, dy, dz))

grid = Grid(shape=(nx, ny, nz), extent=(2., 2., 2.))
u = TimeFunction(name='u', grid=grid, space_order=so)
# init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
u.data[:, :, :, :] = 0
u.data[:, :, :, int(nz/2)] = 1

a = Constant(name='a')
# Create an equation with second-order derivatives
eq = Eq(u.dt, a * u.laplace, subdomain=grid.interior)
stencil = solve(eq, u.forward)
eq_stencil = Eq(u.forward, stencil)

# Create boundary condition expressions
x, y, z = grid.dimensions
t = grid.stepping_dim

print(eq_stencil)

# No BCs
op = Operator([eq_stencil])

# Apply the operator for a number of timesteps
op.apply(time=nt, dt=dt, a=nu)

if args.plot:
    plot_3dfunc(u)

print("Field norm is:", norm(u))

# Reset data for XDSL
u.data[:, :, :, :] = 0
u.data[:, :, :, int(nz/2)] = 1

xdslop = XDSLOperator([eq_stencil])
xdslop.apply(time=nt, dt=dt, a=nu)

if args.plot:
    plot_3dfunc(u)

print("Field norm is:", norm(u))
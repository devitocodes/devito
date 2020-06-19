import numpy as np
import sympy as sp

from devito import *
from devito.types import Lt, Ge, Scalar

import matplotlib.pyplot as plt
from matplotlib import cm

"""
Code for computing the final steady state of the lid driven cavity problem using devito.
Note that transient states are not computed accurately in this formulation.
"""

# physical parameters
rho = Constant(name='rho')
nu = Constant(name='nu')

rho.data = 1.
nu.data = 1./10.

# define spatial mesh
# Size of rectangular domain
Lx = 1
Ly = Lx

# Number of grid points in each direction, including boundary nodes
Nx = 51
Ny = Nx

# hence the mesh spacing
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)

grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly))
gridc = grid
time = grid.time_dim
t = grid.stepping_dim
x, y = grid.dimensions

# time stepping parameters
dt = 1e-3
t_end = 1.
ns = int(t_end/dt)

# Poisson pressure eq. tol
ptol = 1e-3
nit = 50000

# Pressure solve iteration 'dimension'
i = TimeDimension(name='i')
i_space = SteppingDimension(name='i_space', parent=i)
# Pressure solve norm
norm = Scalar(name='norm')

# TODO: Should be a `VectorTimeFunction`.
u = TimeFunction(name='u', grid=grid, space_order=2)
v = TimeFunction(name='v', grid=grid, space_order=2)
p = TimeFunction(name='p', grid=grid, space_order=2)

pi = TimeFunction(name='pi', dimensions=(i_space, x, y),
                  shape=(2, grid.shape[0], grid.shape[1]), space_order=2)

ci = ConditionalDimension(name='ci', parent=i, condition=Ge(norm, ptol))

idims0 = [time, i]
idims1 = [time, i, x, y]
idims2 = [time, i, ci]
idims3 = []

eq_pi0 = Eq(pi.laplace, rho*(1./dt*(u.dxc+v.dyc) - \
    (u.dxc*u.dxc + v.dyc*v.dyc + 2*u.dyc*v.dxc)))
stencil_pi = solve(eq_pi0, pi)

eq_u = Eq(u.dt + u*u.dxc + v*u.dyc - nu*u.laplace, -1./rho*p.forward.dxc)
eq_v = Eq(v.dt + u*v.dxc + v*v.dyc - nu*v.laplace, -1./rho*p.forward.dyc)
stencil_u = solve(eq_u, u.forward)
stencil_v = solve(eq_v, v.forward)

# Main equations

init_norm = Eq(norm, 0)

eq_pi = Eq(pi.forward, stencil_pi, implicit_dims=idims1)

eq_n = Inc(norm, sp.Abs(pi-pi.forward)/(grid.shape[0]*grid.shape[1]),
           implicit_dims=idims0)
break_statement = Inc(norm, -norm, implicit_dims=idims2)

update_p = Eq(p.forward, pi.forward, implicit_dims=idims0)
update_u = Eq(u.forward, stencil_u, implicit_dims=idims3)
update_v = Eq(v.forward, stencil_v, implicit_dims=idims3)

# Create Dirichlet BC expressions for velocity
bc_u = [Eq(u[t+1, x, Ny-1], 1.)]  # top
bc_u += [Eq(u[t+1, 0, y], 0.)]  # left
bc_u += [Eq(u[t+1, Nx-1, y], 0.)]  # right
bc_u += [Eq(u[t+1, x, 0], 0.)]  # bottom
bc_v = [Eq(v[t+1, 0, y], 0.)]  # left
bc_v += [Eq(v[t+1, Nx-1, y], 0.)]  # right
bc_v += [Eq(v[t+1, x, Ny-1], 0.)]  # top
bc_v += [Eq(v[t+1, x, 0], 0.)] # bottom

# Neumann BCs for pressure
bc_p = [Eq(pi[i_space+1, 0, y], pi[i_space+1, 1, y], implicit_dims=idims1)] # left
bc_p += [Eq(pi[i_space+1, Nx-1, y], pi[i_space+1, Nx-2, y], implicit_dims=idims1)] # right
bc_p += [Eq(pi[i_space+1, x, Ny-1], pi[i_space+1, x, Ny-2], implicit_dims=idims1)] # top
bc_p += [Eq(pi[i_space+1, x, 0], pi[i_space+1, x, 1], implicit_dims=idims1)] # bottom

# Pressure only known up to a constant so set to zero at origin
bc_p += [Eq(pi[i_space+1, 0, 0], 0., implicit_dims=idims1)]  # bottom

# Create the operator
exprs = [init_norm, eq_pi] + bc_p + [eq_n, update_p, break_statement, update_u, update_v] + bc_u + bc_v
#op = Operator(exprs, opt='noop') <- For easier to read code
op = Operator(exprs)
op(time_m=0, time_M=ns-1, i_m=0, i_M=nit, dt=dt)

# Check results
Marchi_Re10_u = np.array([[0.0625, -3.85425800e-2],
                          [0.125,  -6.96238561e-2],
                          [0.1875, -9.6983962e-2],
                          [0.25,  -1.22721979e-1],
                          [0.3125, -1.47636199e-1],
                          [0.375,  -1.71260757e-1],
                          [0.4375, -1.91677043e-1],
                          [0.5,    -2.05164738e-1],
                          [0.5625, -2.05770198e-1],
                          [0.625,  -1.84928116e-1],
                          [0.6875, -1.313892353e-1],
                          [0.75,   -3.1879308e-2],
                          [0.8125,  1.26912095e-1],
                          [0.875,   3.54430364e-1],
                          [0.9375,  6.50529292e-1]])

Marchi_Re10_v = np.array([[0.0625, 9.2970121e-2],
                          [0.125, 1.52547843e-1],
                          [0.1875, 1.78781456e-1],
                          [0.25, 1.76415100e-1],
                          [0.3125, 1.52055820e-1],
                          [0.375, 1.121477612e-1],
                          [0.4375, 6.21048147e-2],
                          [0.5, 6.3603620e-3],
                          [0.5625, -5.10417285e-2],
                          [0.625, -1.056157259e-1],
                          [0.6875, -1.51622101e-1],
                          [0.75, -1.81633561e-1],
                          [0.8125, -1.87021651e-1],
                          [0.875, -1.59898186e-1],
                          [0.9375, -9.6409942e-2]])

x_coord = np.linspace(0, Lx, grid.shape[0])
y_coord = np.linspace(0, Ly, grid.shape[1])

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.plot(u.data[-1,int(grid.shape[0]/2),:],y_coord[:])
ax1.plot(Marchi_Re10_u[:,1],Marchi_Re10_u[:,0],'ro')
ax1.set_xlabel('$u$')
ax1.set_ylabel('$y$')
ax1 = fig.add_subplot(122)
ax1.plot(x_coord[:],v.data[-1,:,int(grid.shape[0]/2)])
ax1.plot(Marchi_Re10_v[:,0],Marchi_Re10_v[:,1],'ro')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$v$')

plt.show()

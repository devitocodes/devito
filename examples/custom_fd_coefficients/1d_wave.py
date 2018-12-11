import numpy as np
from devito import Grid, Function, TimeFunction, Eq, Operator, solve
from devito import ConditionalDimension, Constant, first_derivative, second_derivative
from devito import left, right
from math import exp

import matplotlib
import matplotlib.pyplot as plt

def compute_zeta(x,t):
    return 4.0*(x-t)-1.0

def compute_u(x,t):
    u1 = 4.0*compute_zeta(x,t)*(1.0-compute_zeta(x,t))
    u2 = np.zeros(u1.shape)
    return (np.maximum(u1,u2))**(12)

# define spatial mesh
# Size of rectangular domain
Lx = 2
eta = 0.5 # Some value between 0 and 1

# Number of grid points in each direction, including boundary nodes
Nx = 401

# hence the mesh spacing
dx = Lx/(Nx-1)

xg = np.linspace(0,Lx,Nx)

grid = Grid(shape=(Nx), extent=(Lx))
time = grid.time_dim
t = grid.stepping_dim
x = grid.dimensions

#x_t = x*grid.spacing[0]

ug = compute_u(xg,0)

#fig, ax = plt.subplots()
#ax.plot(xg,ug)
#ax.set(xlabel='x', ylabel='u(x,0)')
#ax.grid()
#plt.show()

# time stepping parameters
t_end = 2*(1+eta*dx)
dt = t_end/2000.0
ns = int(t_end/dt)

# Devito computation
u = TimeFunction(name='u', grid=grid, time_order=2, space_order=4, save=ns+1)
u.data[:] = ug[:]

# Main equations
eq = Eq(u.dt2-u.dx2)
stencil = solve(eq, u.forward)

# bc's
bc = [Eq(u[time+1,0], 0.0)]
bc += [Eq(u[time+1,-1], 0.0)]

# IM mods:
def emat(eta):
    E = np.zeros((2,2))
    E[0,0] = -(1.-eta)*(1.0-2.0*eta)/(1.0+eta)/(1.0+2.0*eta)
    E[0,1] = -4.0*(1.0-eta)/(1.0+2.0*eta)
    E[1,0] = -4.0*(2.0-eta)*(1.0-eta)/(1.0+eta)/(1.+2.0*eta)
    E[1,1] = 3.0*(2.0-eta)*(1.0-2.0*eta)/eta/(1.0+2.0*eta)
    return E
E = emat(eta)
bci = [Eq(u[time+1,201], E[0,0]*u[time+1,199]+E[0,1]*u[time+1,200])]
bci += [Eq(u[time+1,202], E[1,0]*u[time+1,199]+E[1,1]*u[time+1,200])]
for j in range(203,Nx+1):
    bci += [Eq(u[time+1,j], 0.0)]
    
# My coeffs
x_coeffs = (x[0], 2, np.array([-1/12, 4/3, -5/2, 4/3, -1/12]))
t_coeffs = (time, 2, np.array([1.0, -2.0, 1.0]))

# Create the operators
op = Operator([Eq(u.forward, stencil)]+bc+bci, t_coeffs, x_coeffs)

op.apply(time_M=ns-1, dt=dt)

#fig, ax = plt.subplots()
#ax.plot(xg,u.data[0,:])
#ax.plot(xg,u.data[600,:])
#ax.set(xlabel='x', ylabel='u(x,2)')
#ax.grid()
#plt.show()

fig, ax = plt.subplots()
ax.plot(xg,u.data[0,:])
ax.plot(xg,u.data[-1,:])
ax.set(xlabel='x', ylabel='u(x,t)')
ax.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(xg[0:201],abs(u.data[-1,0:201]-u.data[0,0:201]))
ax.set(xlabel='x', ylabel='|u(x,2)-u(x,0)|')
ax.grid()
plt.show()

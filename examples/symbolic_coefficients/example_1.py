import numpy as np
from devito import Grid, Function, TimeFunction, Eq, Operator, solve
from devito import ConditionalDimension, Constant, Coefficients
from devito import first_derivative, second_derivative

from devito import left, right
from math import exp

import matplotlib
import matplotlib.pyplot as plt

from sympy import Symbol, IndexedBase, Indexed, Idx, Function

# Remove
from sympy import Wild
from sympy.abc import x, y

# Grid
Lx = 10
Nx = 11
dx = Lx/(Nx-1)

grid = Grid(shape=(Nx), extent=(Lx))
time = grid.time_dim
t = grid.stepping_dim
x = grid.dimensions
h_x = grid.spacing

# time stepping parameters
t_end = 1.0
dt = 0.01
ns = int(t_end/dt)+1

# Devito computation
u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2, save=ns, coefficients='symbolic')
v = TimeFunction(name='v', grid=grid, time_order=2, space_order=2, save=ns, coefficients='standard')

#term = u.dx

#help(term)

# Initalise
u.data[:] = 0.0
v.data[:] = 0.0

# Modified coefficients
#u_x_coeffs = (1, u, x[0], np.array([1.0, -2.0, 1.0]))
#u_t_coeffs = (1, u, time, np.array([1.0, -2.0, 1.0]))

u_x_coeffs = (1, u, x[0], np.array([-0.5, 0.0, 0.5]))
u_t_coeffs = (1, u, time, np.array([-0.5, 0.0, 0.5]))

coeffs=Coefficients(u_x_coeffs,u_t_coeffs)
#coeffs=Coefficients(u_x_coeffs)

# Main equations
eq = Eq(u.dt+u.dx+v.dx, coefficients=coeffs)
#eq = Eq(u.dt+u.dx+v.dx)

print(eq)
#help(eq)

#eq = Eq(u.dt+(u*v+u**2*v[time,x+2*h_x]).dx, coefficients=coeffs)
#eq = Eq(u.dt+(u*v+u**2*v[time,x+2*h_x]).dx)

#print(eq)

#stencil = solve(eq, u.forward)

## bc's
#bc = [Eq(u[time+1,0], 0.0)]
#bc += [Eq(u[time+1,-1], 0.0)]

## My coeffs
#x_coeffs = (x[0], np.array([0.0, 1.0, -2.0, 1.0, 0.0]))
#t_coeffs = (time, np.array([-1/12, 4/3, -5/2, 4/3, -1/12]))

## Create the operators
##op = Operator([Eq(u.forward, stencil)]+bc, x_coeffs, t_coeffs)
##op = Operator([Eq(u.forward, stencil)], t_coeffs)

#op = Operator(Eq(u.forward, stencil))

##print(op.ccode)

#op.apply(time_m=0, time_M=ns-1, dt=dt)

#print(u.data[-1:,0], exp(1.))

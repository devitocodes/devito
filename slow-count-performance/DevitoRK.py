
# coding: utf-8

# In[14]:


import os
import sys

if not "DEVITO_OPENMP" in os.environ or os.environ["DEVITO_OPENMP"] != "1":
    print("*** WARNING: Devito OpenMP environment variable has not been set ***", file=sys.stderr)
os.environ['OMP_NUM_THREADS'] = "8"

from sympy import Matrix
from devito import (Function, TimeFunction, configuration, Operator,
                    second_derivative, first_derivative, Grid)
from sympy import Eq, simplify

def curl(u, dims):
    idx = [((i + 2) % 3, (i + 1) % 3) for i in range(3)]
    return Matrix([first_derivative(u[i], dim=dims[j], order=2) - first_derivative(u[j], dim=dims[i], order=2) for i, j in idx])

# Vector laplacian
def laplacian(u, dims):
    return Matrix([sum([second_derivative(ui, order=2, dim=d) for d in dims]) for ui in u])

# A 3D time function
def MatrixTimeFunction(name, settings):
    return Matrix([TimeFunction(name='{}_{}'.format(name, n), **settings) for n in ['x','y','z']])

# The problem to solve, dm/dt = LLG(m)
def LLG(m, settings, dims):
    c = 2 / (settings.mu0 * settings.Ms)
    e = Matrix(settings.e)
    zeeman = Matrix(settings.H)
    exchange = settings.A * c * laplacian(m, dims)
    anisotropy = settings.K * c * m.dot(e) * e
    dmi = settings.D * c * curl(m, dims)
    heff = zeeman + exchange + anisotropy + dmi
    crossHeff = m.cross(heff)
    LLG = -settings.gamma0 / (1 + settings.alpha**2) * (crossHeff + settings.alpha * m.cross(crossHeff))
    return LLG


# In[15]:


# Convience function that converts a dictionary into a struct
class Struct(object):
    def __init__(self, kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = Struct(v) if isinstance(v, dict) else v

# These material params define the complexity of the LLG as each determines whether a given term is included
MaterialParams = {'A': 1e-11, # Exchange (Align with neighbours) 2nd Order
                  'D': 0, #1.58e-3, # DMI (Anti-align with neighbours) 2nd Order
                  'H': (0.0, 0.0, 0.1e3), # Zeeman (Align with external field) 0th Order
                  'K': 0.1e3, # Anisotropy (Align with material) 0th Order
                  'e': (1, 0, 0), # Anisotropy Axis (unit vector)
                  'alpha': 0.8} # Damping

# Use this one for quick compilation
SuperSimpleMaterialParams = {'A': 0,
                             'D': 0,
                             'H': (0.0, 0.0, 0.0),
                             'K': 0.1e3,
                             'e': (1, 0, 0),
                             'alpha': 0.8}

# Constants and grid settings
SimulationParams = {'Ms': 800000.0,
 'gamma0': 221100.0,
 'mu0': 1.2566370614e-06,
 'grid': {'l': (1.0000000000000001e-07, 5.0000000000000004e-08, 1e-08),
          'n': (100, 50, 10)},
 'time': {'d': 1e-14, 'l': 1e-11, 'n': 1000}}

settings = Struct({**MaterialParams, **SimulationParams})


# In[18]:

print("Generate Stencil")
# Different RK tableaus.
#RKc = [[0], [1]] # Forward Euler
#RKc = [[0, 0], [0.5, 0], [0, 1]] # Midpoint Method
RKc = [[0, 0, 0], [0.5, 0, 0], [2, 1, 0], [1/6, 2/3, 1/6]] # RK3
"""
k_0 = LLG(m_i)
k_1 = LLG(m_i + 0.5 * s * k_0)
k_2 = LLG(m_i + 2 * s * k_0 + s * k_1)
m_(i+1) = m_i + s * (k_0 / 6 + 2 * k_1 / 3 + k_2 / 6)
"""

grid = Grid(shape=settings.grid.n, extent=settings.grid.l)
m = MatrixTimeFunction('m', {"grid":grid, "space_order":2 * (len(RKc) - 1), "time_order":1})
s = grid.time_dim.spacing

# Do the RK cumaltive operation; k_(i+1) = LLG(m + s * sum_0^i(c_j * k_j))
k = []
for c in RKc[:-1]:
    temp = m
    for i, ki in enumerate(k):
        temp += s * c[i] * ki
    k.append(LLG(temp, settings, grid.dimensions))

# Do the final timestep; m_(i+1) = m_i + s * sum_0^i(c_j * k_j)
temp = m
for i, c in enumerate(RKc[-1]):
    temp += s * c * k[i]

stencil = [Eq(mi.forward, ti) for mi, ti in zip(m,temp)]


# In[18]:


print("Generate Operator")
#import time

#print(time.time())
op = Operator(stencil)
#print(op.ccode, file=sys.stderr)
#print(time.time())


# In[19]:

print("Compile")
op.cfunction


# In[21]:


(1513293057.4002063-1513300689.5507386)/60

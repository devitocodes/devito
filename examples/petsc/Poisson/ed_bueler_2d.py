import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm, mmax)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 2D test
# Solving -u_xx - u_yy = f(x,y)
# Dirichlet BCs: u(0,y) = 0, u(1,y)=-e^y, u(x,0) = -x, u(x,1)=-xe
# Manufactured solution: u(x,y) = -xe^(y), with corresponding RHS f(x,y) = xe^(y)
# ref - https://github.com/bueler/p4pdes/blob/master/c/ch6/fish.c

PetscInitialize()

# Subdomains to implement BCs
class SubTop(SubDomain):
    name = 'subtop'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('right', 1)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', 1)}


class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: y}


class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: y}


sub1 = SubTop()
sub2 = SubBottom()
sub3 = SubLeft()
sub4 = SubRight()

subdomains = (sub1, sub2, sub3, sub4)

def exact(x, y):
    return -x*np.float64(np.exp(y))

Lx = np.float64(1.)
Ly = np.float64(1.)

n = 17
h = Lx/(n-1)


grid = Grid(
    shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
)

u = Function(name='u', grid=grid, space_order=2)
f = Function(name='f', grid=grid, space_order=2)
bc = Function(name='bc', grid=grid, space_order=2)

eqn = Eq(-u.laplace, f, subdomain=grid.interior)

tmpx = np.linspace(0, Lx, n).astype(np.float64)
tmpy = np.linspace(0, Ly, n).astype(np.float64)

Y, X = np.meshgrid(tmpx, tmpy)

f.data[:] = X*np.float64(np.exp(Y))

bc.data[0, :] = 0.
bc.data[-1, :] = -np.exp(tmpy)
bc.data[:, 0] = -tmpx
bc.data[:, -1] = -tmpx*np.exp(1)

# # Create boundary condition expressions using subdomains
bcs = [EssentialBC(u, bc, subdomain=sub1)]
bcs += [EssentialBC(u, bc, subdomain=sub2)]
bcs += [EssentialBC(u, bc, subdomain=sub3)]
bcs += [EssentialBC(u, bc, subdomain=sub4)]

exprs = [eqn] + bcs
petsc = petscsolve(
    exprs, target=u,
    solver_parameters={'ksp_rtol': 1e-12, 'ksp_type': 'cg', 'pc_type': 'none'},
    options_prefix='poisson_2d',
    constrain_bcs=True
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    summary = op.apply()
    # print(op.arguments())


# print(op.ccode)
# iters = summary.petsc[('section0', 'poisson_2d')].KSPGetIterationNumber

u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X, Y)
print(u_exact)

diff = Function(name='diff', grid=grid, space_order=2)
diff.data[:] = u_exact.data[:] - u.data[:]

# # Compute infinity norm using numpy
# # TODO: Figure out how to compute the infinity norm using Devito
infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
print(f"Infinity Norm={infinity_norm}")

# # Compute discrete L2 norm (RMS error)
n_interior = np.prod([s - 1 for s in grid.shape])
discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
print(f"Discrete L2 Norm={discrete_l2_norm}")

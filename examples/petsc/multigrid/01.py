import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain, norm)

from devito.petsc import petscsolve, EssentialBC, GridHierarchy
from devito.petsc.initialize import PetscInitialize

import matplotlib.pyplot as plt

configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 1D test
# Solving -u.laplace = f(x)
# Dirichlet BCs: u(0) = -1, u(1) = -e
# Manufactured solution: u(x) = -e^(x), with corresponding RHS f(x) = e^(x)
# ref - https://github.com/bueler/p4pdes/blob/master/c/ch6/fish.c

PetscInitialize()

# Subdomains to implement BCs
class SubLeft(SubDomain):
    name = 'subleft'

    def define(self, dimensions):
        x, = dimensions
        return {x: ('left', 1)}


class SubRight(SubDomain):
    name = 'subright'

    def define(self, dimensions):
        x, = dimensions
        return {x: ('right', 1)}


sub1 = SubLeft()
sub2 = SubRight()

subdomains = (sub1, sub2,)


def exact(x):
    return -np.float64(np.exp(x))


Lx = np.float64(1.)

n = 33

grid = Grid(
    shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
)
hierarchy = GridHierarchy(grid, nlevels=3)

u = Function(name='u', grid=grid, space_order=2)
f = Function(name='f', grid=grid, space_order=2)
bc = Function(name='bc', grid=grid, space_order=2)

eqn = Eq(-u.laplace, f, subdomain=grid.interior)

X = np.linspace(0, Lx, n).astype(np.float64)
f.data[:] = np.float64(np.exp(X))

bc.data[0] = -np.float64(1.0)  # u(0) = -1
bc.data[-1] = -np.float64(np.exp(1.0))  # u(1) = -e

# Create boundary condition expressions using subdomains
bcs = [EssentialBC(u, bc, subdomain=sub1)]
bcs += [EssentialBC(u, bc, subdomain=sub2)]

exprs = [eqn] + bcs
petsc = petscsolve(
    exprs, target=u,
    hierarchy=hierarchy,
    solver_parameters={
        'ksp_type': 'cg',
        'pc_type': 'mg',
        'snes_type': 'ksponly',
        'mg_levels_ksp_type': 'chebyshev',
        'mg_levels_pc_type': 'jacobi',
        'mg_coarse_ksp_type': 'gmres',
        'mg_coarse_pc_type': 'none',
    },
    options_prefix='poisson_1d',
)

with switchconfig(log_level='DEBUG'):
    op = Operator(petsc, language='petsc')
    print(op.ccode)
    summary = op.apply()

iters = summary.petsc[('section0', 'poisson_1d')].KSPGetIterationNumber

u_exact = Function(name='u_exact', grid=grid, space_order=2)
u_exact.data[:] = exact(X)

diff = Function(name='diff', grid=grid, space_order=2)
diff.data[:] = u_exact.data[:] - u.data[:]

# Compute infinity norm using numpy
# TODO: Figure out how to compute the infinity norm using Devito
infinity_norm = np.linalg.norm(diff.data[:].ravel(), ord=np.inf)
print(f"Infinity norm of error: {infinity_norm}")

# Compute discrete L2 norm (RMS error)
n_interior = np.prod([s - 1 for s in grid.shape])
discrete_l2_norm = norm(diff) / np.sqrt(n_interior)
print(f"Discrete L2 norm of error: {discrete_l2_norm}")
    

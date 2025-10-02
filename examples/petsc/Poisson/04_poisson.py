import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain)

from devito.petsc import petscsolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# 2D test
# Solving u.laplace = 0
# Dirichlet BCs.
# ref - https://www.scirp.org/journal/paperinformation?paperid=113731#f2
# example 2 -> note they wrote u(x,1) bc wrong, it should be u(x,y) = e^-pi*sin(pix)


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


def analytical(x, y):
    return np.float64(np.exp(-y*np.pi)) * np.float64(np.sin(np.pi*x))


Lx = np.float64(1.)
Ly = np.float64(1.)

n_values = [10, 30, 50, 70, 90, 110]
dx = np.array([Lx/(n-1) for n in n_values])
errors = []


for n in n_values:
    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=2)
    rhs = Function(name='rhs', grid=grid, space_order=2)

    eqn = Eq(rhs, u.laplace, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)

    Y, X = np.meshgrid(tmpx, tmpy)

    rhs.data[:] = 0.

    bcs = Function(name='bcs', grid=grid, space_order=2)

    bcs.data[:, 0] = np.sin(np.pi*tmpx)
    bcs.data[:, -1] = np.exp(-np.pi)*np.sin(np.pi*tmpx)
    bcs.data[0, :] = 0.
    bcs.data[-1, :] = 0.

    # # Create boundary condition expressions using subdomains
    bc_eqns = [EssentialBC(u, bcs, subdomain=sub1)]
    bc_eqns += [EssentialBC(u, bcs, subdomain=sub2)]
    bc_eqns += [EssentialBC(u, bcs, subdomain=sub3)]
    bc_eqns += [EssentialBC(u, bcs, subdomain=sub4)]

    exprs = [eqn]+bc_eqns
    petsc = petscsolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-6})

    with switchconfig(language='petsc'):
        op = Operator(petsc)
        op.apply()

    u_exact = analytical(X, Y)

    diff = u_exact[1:-1] - u.data[1:-1]
    error = np.linalg.norm(diff) / np.linalg.norm(u_exact[1:-1])
    errors.append(error)

slope, _ = np.polyfit(np.log(dx), np.log(errors), 1)


assert slope > 1.9
assert slope < 2.1

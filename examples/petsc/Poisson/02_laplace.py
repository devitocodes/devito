import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, SubDomain,
                    configuration, switchconfig)
from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

PetscInitialize()

# Laplace equation, solving phi.laplace = 0

# Constant Dirichlet BCs:
# phi(x, 0) = 0
# phi(0, y) = 0
# phi(1, y) = 0
# phi(x, 1) = f(x) = sin(pi*x)

# The analytical solution is:
# phi(x, y) = sinh(pi*y)*sin(pi*x)/sinh(pi)


# Subdomains to implement BCs
class SubTop(SubDomain):
    name = 'subtop'

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('right', 1)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('left', 1)}


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


Lx = np.float64(1.)
Ly = np.float64(1.)


def analytical(x, y, Lx, Ly):
    tmp = np.float64(np.pi)/Lx
    numerator = np.float64(np.sinh(tmp*y)) * np.float64(np.sin(tmp*x))
    return numerator / np.float64(np.sinh(tmp*Ly))


n_values = list(range(13, 174, 10))
dx = np.array([Lx/(n-1) for n in n_values])
errors = []


for n in n_values:

    grid = Grid(
        shape=(n, n), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
    )

    phi = Function(name='phi', grid=grid, space_order=2, dtype=np.float64)
    rhs = Function(name='rhs', grid=grid, space_order=2, dtype=np.float64)

    phi.data[:] = np.float64(0.0)
    rhs.data[:] = np.float64(0.0)

    eqn = Eq(rhs, phi.laplace, subdomain=grid.interior)

    tmpx = np.linspace(0, Lx, n).astype(np.float64)
    tmpy = np.linspace(0, Ly, n).astype(np.float64)
    Y, X = np.meshgrid(tmpx, tmpy)

    # Create boundary condition expressions using subdomains
    bc_func = Function(name='bcs', grid=grid, space_order=2, dtype=np.float64)
    bc_func.data[:] = np.float64(0.0)
    bc_func.data[:, -1] = np.float64(np.sin(tmpx*np.pi))

    bcs = [EssentialBC(phi, bc_func, subdomain=sub1)]  # top
    bcs += [EssentialBC(phi, bc_func, subdomain=sub2)]  # bottom
    bcs += [EssentialBC(phi, bc_func, subdomain=sub3)]  # left
    bcs += [EssentialBC(phi, bc_func, subdomain=sub4)]  # right

    exprs = [eqn] + bcs
    petsc = PETScSolve(exprs, target=phi, solver_parameters={'ksp_rtol': 1e-8})

    with switchconfig(language='petsc'):
        op = Operator(petsc)
        op.apply()

    phi_analytical = analytical(X, Y, Lx, Ly)

    diff = phi_analytical[1:-1, 1:-1] - phi.data[1:-1, 1:-1]
    error = np.linalg.norm(diff) / np.linalg.norm(phi_analytical[1:-1, 1:-1])
    errors.append(error)

slope, _ = np.polyfit(np.log(dx), np.log(errors), 1)

assert slope > 1.9
assert slope < 2.1

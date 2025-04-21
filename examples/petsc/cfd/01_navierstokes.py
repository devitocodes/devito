import os
import numpy as np

from devito import (Grid, TimeFunction, Function, Constant, Eq,
                    Operator, norm, SubDomain, switchconfig, configuration)
from devito.symbolics import retrieve_functions, INT

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


PetscInitialize()

# Chorin's projection method
# Explicit time-stepping

# Physical parameters
rho = Constant(name='rho', dtype=np.float64)
nu = Constant(name='nu', dtype=np.float64)

rho.data = np.float64(1.)
nu.data = np.float64(1./10.)

Lx = 1.
Ly = Lx

# Number of grid points in each direction
nx = 41
ny = 41

# mesh spacing
dx = Lx/(nx-1)
dy = Ly/(ny-1)
so = 2


# Use subdomains just for pressure field for now
class SubTop(SubDomain):
    name = 'subtop'

    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('right', self.S_O//2)}


class SubBottom(SubDomain):
    name = 'subbottom'

    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 1, 1), y: ('left', self.S_O//2)}


class SubLeft(SubDomain):
    name = 'subleft'

    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', self.S_O//2), y: ('middle', 1, 1)}


class SubRight(SubDomain):
    name = 'subright'

    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', self.S_O//2), y: ('middle', 1, 1)}


class SubPointBottomLeft(SubDomain):
    name = 'subpointbottomleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('left', 1)}


class SubPointBottomRight(SubDomain):
    name = 'subpointbottomright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('left', 1)}


class SubPointTopLeft(SubDomain):
    name = 'subpointtopleft'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: ('right', 1)}


class SubPointTopRight(SubDomain):
    name = 'subpointtopright'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: ('right', 1)}


def neumann_bottom(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] < 0:
            if f.name == 'pn1':
                mapper.update({f: f.subs({yind: INT(abs(yind))})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_top(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] > 0:
            # Symmetric mirror
            tmp = y - INT(abs(y.symbolic_max - yind))
            if f.name == 'pn1':
                mapper.update({f: f.subs({yind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_left(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] < 0:
            # Symmetric mirror
            # Substitute where index is negative for +ve
            # where index is positive
            if f.name == 'pn1':
                mapper.update({f: f.subs({xind: INT(abs(xind))})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


def neumann_right(eq, subdomain):
    lhs, rhs = eq.evaluate.args

    # Get horizontal subdimension and its parent
    xfs = subdomain.dimensions[0]
    x = xfs.parent

    # Functions present in stencil
    funcs = retrieve_functions(lhs-rhs)

    mapper = {}
    for f in funcs:
        # Get the x index
        xind = f.indices[-2]
        if (xind - x).as_coeff_Mul()[0] > 0:
            tmp = x - INT(abs(x.symbolic_max - xind))
            if f.name == 'pn1':
                mapper.update({f: f.subs({xind: tmp})})

    return Eq(lhs.subs(mapper), rhs.subs(mapper), subdomain=subdomain)


sub1 = SubTop(so)
sub2 = SubBottom(so)
sub3 = SubLeft(so)
sub4 = SubRight(so)
sub5 = SubPointBottomLeft()
sub6 = SubPointBottomRight()
sub7 = SubPointTopLeft()
sub8 = SubPointTopRight()

subdomains = (sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8)

grid = Grid(
    shape=(nx, ny), extent=(Lx, Ly), subdomains=subdomains, dtype=np.float64
)
time = grid.time_dim
t = grid.stepping_dim
x, y = grid.dimensions

# time stepping parameters
dt = 1e-3
t_end = 1.
ns = int(t_end/dt)

u1 = TimeFunction(name='u1', grid=grid, space_order=2, dtype=np.float64)
v1 = TimeFunction(name='v1', grid=grid, space_order=2, dtype=np.float64)
pn1 = Function(name='pn1', grid=grid, space_order=2, dtype=np.float64)

pn1.data[:] = 0.

eq_pn1 = Eq(pn1.laplace, rho*(1./dt*(u1.forward.dxc+v1.forward.dyc)),
            subdomain=grid.interior)


bc_pn1 = [neumann_top(eq_pn1, sub1)]
bc_pn1 += [neumann_bottom(eq_pn1, sub2)]
bc_pn1 += [neumann_left(eq_pn1, sub3)]
bc_pn1 += [neumann_right(eq_pn1, sub4)]
bc_pn1 += [EssentialBC(pn1, 0., subdomain=sub5)]
bc_pn1 += [neumann_right(neumann_bottom(eq_pn1, sub6), sub6)]
bc_pn1 += [neumann_left(neumann_top(eq_pn1, sub7), sub7)]
bc_pn1 += [neumann_right(neumann_top(eq_pn1, sub8), sub8)]


eqn_p = PETScSolve([eq_pn1]+bc_pn1, pn1)

eq_u1 = Eq(u1.dt + u1*u1.dxc + v1*u1.dyc, nu*u1.laplace)
eq_v1 = Eq(v1.dt + u1*v1.dxc + v1*v1.dyc, nu*v1.laplace)

update_u = Eq(u1.forward, u1.forward - (dt/rho)*(pn1.dxc),
              subdomain=grid.interior)

update_v = Eq(v1.forward, v1.forward - (dt/rho)*(pn1.dyc),
              subdomain=grid.interior)

# TODO: Can drop due to initial guess CB
u1.data[0, :, -1] = np.float64(1.)
u1.data[1, :, -1] = np.float64(1.)


# TODO: Don't need both sets of bcs, can reuse petsc ones
# Create Dirichlet BC expressions for velocity
bc_u1 = [Eq(u1[t+1, x, ny-1], 1.)]  # top
bc_u1 += [Eq(u1[t+1, 0, y], 0.)]  # left
bc_u1 += [Eq(u1[t+1, nx-1, y], 0.)]  # right
bc_u1 += [Eq(u1[t+1, x, 0], 0.)]  # bottom
bc_v1 = [Eq(v1[t+1, 0, y], 0.)]  # left
bc_v1 += [Eq(v1[t+1, nx-1, y], 0.)]  # right
bc_v1 += [Eq(v1[t+1, x, ny-1], 0.)]  # top
bc_v1 += [Eq(v1[t+1, x, 0], 0.)]  # bottom

# Create Dirichlet BC expressions for velocity
bc_petsc_u1 = [EssentialBC(u1.forward, 1., subdomain=sub1)]  # top
bc_petsc_u1 += [EssentialBC(u1.forward, 0., subdomain=sub3)]  # left
bc_petsc_u1 += [EssentialBC(u1.forward, 0., subdomain=sub4)]  # right
bc_petsc_u1 += [EssentialBC(u1.forward, 0., subdomain=sub2)]  # bottom
bc_petsc_v1 = [EssentialBC(v1.forward, 0., subdomain=sub3)]  # left
bc_petsc_v1 += [EssentialBC(v1.forward, 0., subdomain=sub4)]  # right
bc_petsc_v1 += [EssentialBC(v1.forward, 0., subdomain=sub1)]  # top
bc_petsc_v1 += [EssentialBC(v1.forward, 0., subdomain=sub2)]  # bottom

tentu = PETScSolve([eq_u1]+bc_petsc_u1, u1.forward)
tentv = PETScSolve([eq_v1]+bc_petsc_v1, v1.forward)

exprs = tentu + tentv + eqn_p + [update_u, update_v] + bc_u1 + bc_v1

with switchconfig(language='petsc'):
    op = Operator(exprs)
    op.apply(time_m=0, time_M=ns-1, dt=dt)

u1_norm = norm(u1)
v1_norm = norm(v1)
p1_norm = norm(pn1)


# TODO: change these norm checks to array checks (use paper)
assert np.isclose(u1_norm, 13.966067703420883, atol=0, rtol=1e-7)
assert np.isclose(v1_norm, 7.9575677674738285, atol=0, rtol=1e-7)
assert np.isclose(p1_norm, 36.46263134701362, atol=0, rtol=1e-7)

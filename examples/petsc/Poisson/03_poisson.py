import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, switchconfig,
                    configuration, SubDomain)

from devito.petsc import PETScSolve, EssentialBC
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'

# 1D test
# ref - An efficient implementation of fourth-order compact finite
# difference scheme for Poisson equation with Dirichlet
# boundary conditions
# https://pdf.sciencedirectassets.com/271503/1-s2.0-S0898122116X00090/1-s2.0-S0898122116300761/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjENH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIH%2F3tmUOBoYeSj%2FmE47N2yF5b5IvdQohX7JWYGRZhNFTAiAX0P7PnDVBy2D3pTjsZRFgFEh7nVzV8oxoUFevx%2FucWSqzBQhZEAUaDDA1OTAwMzU0Njg2NSIMHBuo2q1ii2daxqQiKpAFdh12uILywsya5sCFqTsXwUdb%2F4lvp5uZIigTrb18tjM8cPru5xrZDgDVYVIlT4G6L1SE05FkjWKSQ7AO24wec3y2bNKGgpC%2FPFbEmTv6CgpR%2FjoSpToblGLiOqTkUgICSK3EoMdah%2BWl552nO0Ajdwuor0brGfDM7C2fgH1FqM%2BLyJ2do33rYFjGAswzZsQOGcf%2BChdkaKA9bxvfgE2%2Bukf7RcYDsGteSEX5Zb9XoyvMheiMUZoZk7KVPWjj3JORx9qetLs9LkpPO3IU%2BqPxtM7Vt3BnEnXR9gQ2bnL%2FtcT%2FcvsZ7a8AdiU1j8%2F%2Fxi9nBgPow0MQTmaoe9n67XRS0BVE7wAWldDb2qdZuOfwYl%2F2iG78mMTn%2FC4YcOCezc4nUT9fTcTcv3wKZzA%2Bkh8Z%2BXvdTcdADCKdVaIXLylqlhEmBlwua4cGjBG0RbpvGa%2FOBk6CbZLpn7%2FLawxsVZ1U1ksGd8HGJ%2FGMYDOauM%2FhRGNWRFsXnn%2BsrPhaJ3SoirVeV3q9JVrjGT6%2FUT3W9qIDtdPP4MJae5mp6TG5fusJjkCLxLTbeXF0%2FhbwEnAA54uj3jpTsh7rXVDB%2B8skGSdMhIITz3%2ByS%2BdMqt7iEgFOWqYXGwgXLGbOqyGGz2ikth4cs1FMT4sYrA066%2BcMkE9q3l3bsFZHQMw13UPgJQp2f69JIzgHbZ%2FoCkdDYNxUutRhZ6cMitSLrIGtcAa7p%2Fevtejnw5eTz20kLNAxjB3CMUuS1H5qhxb6cSmxneilYH1WINNPjCrDPCJ3FxlKtCJo4QzIfIKogegd%2B44T78fQzt8RP7LfA%2FzjITD9bdiCYW0f81Q3O8zzL7l7RtfnLfYXAuTFh9GtAdE8D6b4F2pnXkMwrfCCwAY6sgG4%2BnyhdUNH%2FhdcK7GZ56erHPDOYF04vpG2hZy26v7cSnA3Xb7zrqVzkLxPdyAViJnMjzV1c8itVIHgnkLuA0C%2FPJrp3RPy0ivl9dofnd%2FLtoBkoBadnTgx2f7x4SZ62bdbWk5DJ%2FavMuOajJ%2F4tl9%2F7%2FLWoyi92xH2ZCvnT4wIIakx9ODzn2dRwSYwP20omrw5oAHK8KfXr39zDhQcs6FZMnWqYVxGlKHy0XIqJY8mTLeE&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250417T092301Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXYYPN3VY%2F20250417%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=bfaba1511f86b81dca784d137decbc826e87f24f30b4589d02ca67033b886446&hash=5e1a53dc45c46e59555d516468c6e00966f056dd50abcad3b239f603507d92a7&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0898122116300761&tid=spdf-f29c66e1-0a20-4e85-99d8-adbf3bfe5f8e&sid=68bdc92a37ea6249ef2b6425bac44510ce06gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1d035b54560505005402&rr=931adc7ff9ea1b7e&cc=gb

# Solving -u.laplace = pi^2 * sin(pix), 0<x<2
# Constant zero Dirichlet BCs.
# u(0) = 0, u(2) = 0

# exact solution is u_exact(x) = sin(pix)

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


def analytical(x):
    return np.float64(np.sin(np.pi*x))


Lx = np.float64(2.)

n_values = list(range(13, 174, 10))
dx = np.array([Lx/(n-1) for n in n_values])
errors = []


for n in n_values:

    grid = Grid(
        shape=(n,), extent=(Lx,), subdomains=subdomains, dtype=np.float64
    )

    u = Function(name='u', grid=grid, space_order=2, dtype=np.float64)
    rhs = Function(name='rhs', grid=grid, space_order=2, dtype=np.float64)

    eqn = Eq(rhs, -u.laplace, subdomain=grid.interior)

    X = np.linspace(0, Lx, n).astype(np.float64)

    rhs.data[:] = np.float64(np.pi**2) * np.sin(np.pi*X)

    # # Create boundary condition expressions using subdomains
    bcs = [EssentialBC(u, np.float64(0.), subdomain=sub1)]
    bcs += [EssentialBC(u, np.float64(0.), subdomain=sub2)]

    exprs = [eqn] + bcs
    petsc = PETScSolve(exprs, target=u, solver_parameters={'ksp_rtol': 1e-7})

    with switchconfig(language='petsc'):
        op = Operator(petsc)
        op.apply()

    u_exact = analytical(X)

    diff = u_exact[1:-1] - u.data[1:-1]
    error = np.linalg.norm(diff) / np.linalg.norm(u_exact[1:-1])
    errors.append(error)

slope, _ = np.polyfit(np.log(dx), np.log(errors), 1)

assert slope > 1.9
assert slope < 2.1

import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, configuration,
                    switchconfig)
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# Initialize petsctools
# petsctools.init()

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--nx', type=int, default=11)
parser.add_argument('--ny', type=int, default=11)

args, unknown = parser.parse_known_args()

PetscInitialize(unknown)

grid = Grid(shape=(args.nx, args.ny), extent=(2., 2.), dtype=np.float64)

u = Function(name='u', grid=grid, dtype=np.float64, space_order=2)
v = Function(name='v', grid=grid, dtype=np.float64, space_order=2)

v.data[:] = 5.0

eq = Eq(v, u.laplace, subdomain=grid.interior)

solver = PETScSolve([eq], u)

with switchconfig(language='petsc', opt='noop'):
    op = Operator(solver)
    op.apply()
    # print(op.ccode)





grid = Grid(shape=(11, 11), dtype=np.float64)
functions = [Function(name=n, grid=grid, space_order=2)
                for n in ['e', 'f']]
e, f = functions

eq = Eq(e.laplace, f)

petsc = PETScSolve(eq, target=e, solver_parameters={'snes_view': None})

with switchconfig(language='petsc'):
    op = Operator(petsc)
    op.apply()

assert 'PetscCall(SetPetscOption("-snes_view",NULL));' \
    in str(op._func_table['SetPetscOptions0'].root)

from devito.petsc.solver_parameters import linear_solve_defaults



from IPython import embed; embed()




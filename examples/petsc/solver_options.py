import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, configuration,
                    switchconfig)
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize
import petsctools
from petsctools import get_commandline_options
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

petsc = PETScSolve([eq], u)

with switchconfig(language='petsc'):
    op = Operator(petsc)
    op.apply()

# print(op.ccode)

# print(grid.shape)


import sys
petsctools.options._commandline_options = sys.argv[1:]
tmp = get_commandline_options()
print("Command line options:", tmp)



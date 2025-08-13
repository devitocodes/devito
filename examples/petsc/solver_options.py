import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, configuration,
                    switchconfig, Constant)
from devito.tools import frozendict
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# Initialize petsctools
# petsctools.init()

# from argparse import ArgumentParser

# parser = ArgumentParser()

# parser.add_argument('--nx', type=int, default=11)
# parser.add_argument('--ny', type=int, default=11)

# args, unknown = parser.parse_known_args()

# PetscInitialize(unknown)

# grid = Grid(shape=(args.nx, args.ny), extent=(2., 2.), dtype=np.float64)

# u = Function(name='u', grid=grid, dtype=np.float64, space_order=2)
# v = Function(name='v', grid=grid, dtype=np.float64, space_order=2)

# v.data[:] = 5.0

# eq = Eq(v, u.laplace, subdomain=grid.interior)

# solver = PETScSolve([eq], u)

# with switchconfig(language='petsc', opt='noop'):
#     op = Operator(solver)
#     # op.apply()
#     # print(op.ccode)


PetscInitialize()


grid = Grid(shape=(11, 11), dtype=np.float64)
functions = [Function(name=n, grid=grid, space_order=2)
                for n in ['e', 'f']]
e, f = functions

eq = Eq(e.laplace, f)

params1 = {'ksp_view': None, 'ksp_rtol': 1e-15}
params2 = {'ksp_rtol': 1e-12, 'ksp_view': None}
petsc1 = PETScSolve(eq, target=e, solver_parameters=params1, options_prefix='pde1')
petsc2 = PETScSolve(eq, target=e, solver_parameters=params2, options_prefix='pde2')


# frozen1 = frozendict(params1)
# frozen2 = frozendict(params2)

# assert hash(frozen1) == hash(frozen2)

# from IPython import embed; embed()

with switchconfig(language='petsc', log_level='DEBUG'):

    op1 = Operator([petsc1])
    # op2 = Operator(petsc2)
    summary1 = op1.apply()
    # summary2 = op2.apply()
    
    # print(op1.ccode)
    # print(op2.ccode)

petsc_summary = summary1.petsc



# from IPython import embed; embed()

# print(petsc1.rhs._solver_parameters)
# print(petsc2.rhs._solver_parameters)
# print(petsc1.rhs._solver_parameters == petsc2.rhs._solver_parameters)
# print(petsc1.rhs.expr == petsc2.rhs.expr)
# print(petsc1.rhs._user_prefix == petsc2.rhs._user_prefix)
# print(petsc1.rhs == petsc2.rhs)



# from devito.petsc.types import SolveExpr

# tmp1 = SolveExpr(e+f, user_prefix='pde1')
# tmp2 = SolveExpr(e+f, user_prefix='pde2')

# print(hash(tmp1))
# print(hash(tmp2))

# tmp1 == tmp2

# from devito.petsc.types import LinearSolveExpr
# petsc1 = LinearSolveExpr(
#             (e, f),
#             solver_parameters={'ksp_view': None, 'ksp_rtol': 1e-15},
#         )

# petsc2 = LinearSolveExpr(
#             (e, f),
#             solver_parameters={'ksp_view': None, 'ksp_rtol': 1e-15},
#         )





    # print(op.ccode)
#
# assert 'PetscCall(SetPetscOption("-snes_view",NULL));' \
#     in str(op._func_table['SetPetscOptions0'].root)

# from devito.petsc.solver_parameters import linear_solve_defaults



# from IPython import embed; embed()


# grid = Grid(shape=(11, 11), dtype=np.float64)
# e, f, g, h = [
#     Function(name=n, grid=grid, space_order=2)
#     for n in ['e', 'f', 'g', 'h']
# ]


# eq1 = Eq(h.laplace, f)

# solver_parameters = {
#     'ksp_rtol': 1e-12,
#     'ksp_atol': 1e-20,
#     'ksp_divtol': 1e3,
#     'ksp_max_it': 100,
#     'ksp_view': None,

# }
# solver = PETScSolve(
#     eq1, target=h, solver_parameters=solver_parameters
# )

# with switchconfig(language='petsc', log_level='DEBUG'):
#     op2 = Operator(solver)
#     summary2 = op2.apply()
#     # print(op.ccode)


# linsolveexpr1 = petsc.rhs
# linsolveexpr2 = solver.rhs

# petsc_summary1 = summary1.petsc
# petsc_summary2 = summary2.petsc

# infos1 = petsc_summary1.petscinfos
# infos2 = petsc_summary2.petscinfos


# op_args1 = op1.arguments()
# op_args2 = op2.arguments()



# # from IPython import embed; embed()

# # entry1 = petsc_summary1.get_entry('section0', 'devito_0')
# # entry2 = petsc_summary2.get_entry('section0', 'devito_1')

# # tols1 = entry1.KSPGetTolerances
# # tols2 = entry2.KSPGetTolerances

# from IPython import embed; embed()

# # assert tols2['rtol'] == solver_parameters['ksp_rtol']
# # assert tols2['abstol'] == solver_parameters['ksp_atol']
# # assert tols2['dtol'] == solver_parameters['ksp_divtol']
# # assert tols2['maxits'] == solver_parameters['ksp_max_it']



# # from IPython import embed; embed()



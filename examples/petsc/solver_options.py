import os
import numpy as np

from devito import (Grid, Function, Eq, Operator, configuration,
                    switchconfig)
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize
import petsctools
from petsctools import get_commandline_options, OptionsManager, flatten_parameters
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

solver = PETScSolve([eq], u, solver_parameters={'ksp_rtol': 1e-8}, options_prefix='poisson')

with switchconfig(language='petsc'):
    op = Operator(solver)
    op.apply()
    print(op.ccode)


# import sys
# petsctools.options._commandline_options = sys.argv[1:]
# tmp = get_commandline_options()
# print("Command line options:", tmp)



# class DevitoOptionsManager(OptionsManager):
#     """
#     """
#     def __init__(self, parameters, options_prefix):
#         if parameters is None:
#             parameters = {}
#         else:
#             # Convert nested dicts
#             parameters = flatten_parameters(parameters)
#         if options_prefix is None:
#             self.options_prefix = "firedrake_%d_" % next(self.count)
#             self.parameters = parameters
#             self.to_delete = set(parameters)
#         else:
#             if len(options_prefix) and not options_prefix.endswith("_"):
#                 options_prefix += "_"
#             self.options_prefix = options_prefix
#             # Remove those options from the dict that were passed on
#             # the commandline.
#             self.parameters = {
#                 k: v
#                 for k, v in parameters.items()
#                 if options_prefix + k not in get_commandline_options()
#             }
#             self.to_delete = set(self.parameters)
#             # Now update parameters from options, so that they're
#             # available to solver setup (for, e.g., matrix-free).
#             # Can't ask for the prefixed guy in the options object,
#             # since that does not DTRT for flag options.
#             # for k, v in self.options_object.getAll().items():
#             #     if k.startswith(self.options_prefix):
#             #         self.parameters[k[len(self.options_prefix):]] = v

#             for k, v in get_commandline_options():
#                 if k.startswith(self.options_prefix):
#                     self.parameters[k[len(self.options_prefix):]] = v
#         self._setfromoptions = False



# options_manager = DevitoOptionsManager(solver.rhs.solver_parameters, solver.rhs.options_prefix)


# nested = {"ksp_type": "cg",
#           "pc_type": "fieldsplit",
#           "fieldsplit_0": {"ksp_type": "gmres",
#                            "pc_type": "hypre",
#                            "ksp_rtol": 1e-5},
#           "fieldsplit_1": {"ksp_type": "richardson",
#                            "pc_type": "ilu"}}

# tmp = flatten_parameters(nested)
# # from IPython import embed; embed()

# # convert all values into strings

# tmp_str = {k: str(v) for k, v in tmp.items()}





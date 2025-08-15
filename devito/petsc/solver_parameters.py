from petsctools import flatten_parameters
import itertools

"""
Parameter descriptions:
- 'ksp_type':    Specifies the Krylov method (e.g., 'gmres', 'cg').
- 'pc_type':     Specifies the preconditioner type (e.g., ...).
- 'ksp_rtol':    Relative convergence tolerance for KSP solvers.
- 'ksp_atol':    Absolute convergence tolerance for KSP solvers.
- 'ksp_divtol':  Divergence tolerance, amount residual norm can increase before
                `KSPConvergedDefault()` concludes that the method is diverging.
- 'ksp_max_it':  Maximum number of KSP iterations to use.
- 'snes_type':   Type of SNES solver; 'ksponly' is used for linear solves.

References:
- KSP types: https://petsc.org/main/manualpages/KSP/KSPType/
- PC types: https://petsc.org/main/manualpages/PC/PCType/
- KSP tolerances: https://petsc.org/main/manualpages/KSP/KSPSetTolerances/
- SNES type: https://petsc.org/main/manualpages/SNES/SNESType/
"""


# NOTE: Will be extended, the default preconditioner is not going to be 'none'
base_solve_defaults = {
    'ksp_type': 'gmres',
    'pc_type': 'none',
    'ksp_rtol': 1.e-5,
    'ksp_atol': 1.e-50,
    'ksp_divtol': 1e5,
    'ksp_max_it': int(1e4)
}


# Specific to linear solvers
linear_solve_defaults = {
    'snes_type': 'ksponly',
    **base_solve_defaults,
}


def linear_solver_parameters(solver_parameters):
    # Flatten parameters to support nested dictionaries
    flattened = flatten_parameters(solver_parameters or {})
    processed = linear_solve_defaults.copy()
    processed.update(flattened)
    return processed


_options_prefix_counter = itertools.count()


def format_options_prefix(options_prefix):
    # NOTE: Modified from the `OptionsManager` inside petsctools
    if options_prefix is None:
        options_prefix = f"devito_{next(_options_prefix_counter)}_"
    else:
        if len(options_prefix) and not options_prefix.endswith("_"):
            options_prefix += "_"
    return options_prefix

import itertools

from petsctools import flatten_parameters


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
    if not options_prefix:
        return f"devito_{next(_options_prefix_counter)}_"

    if options_prefix.endswith("_"):
        return options_prefix
    return f"{options_prefix}_"

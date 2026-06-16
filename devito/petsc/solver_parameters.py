import itertools

try:
    from petsctools import flatten_parameters
except ImportError:
    # TODO: drop
    def flatten_parameters():
        raise ImportError("petsctools is not installed")

# NOTE: Will be extended, the default preconditioner is not going to be 'none'
base_solve_defaults = {
    'ksp_type': 'gmres',
    'pc_type': 'none',
    'ksp_rtol': 1.e-5,
    'ksp_atol': 1.e-50,
    'ksp_divtol': 1e5,
    'ksp_max_it': int(1e4)
}


# Geometric multigrid solver defaults - check with PETSc defaults
mg_solve_defaults = {
    'mg_levels_ksp_type': 'richardson',
    'mg_levels_pc_type': 'jacobi',
    'mg_levels_ksp_max_it': 2,
    'mg_coarse_ksp_type': 'preonly',
    # TODO: in petsc this is 'lu' but don't currently support explicit matrix assembly
    'mg_coarse_pc_type': 'none',
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
    if flattened.get('pc_type') == 'mg':
        processed.update(mg_solve_defaults)
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

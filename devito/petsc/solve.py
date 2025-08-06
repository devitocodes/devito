from devito.types.equation import PetscEq
from devito.tools import as_tuple
from devito.petsc.types import (LinearSolveExpr, PETScArray, DMDALocalInfo,
                                FieldData, MultipleFieldData, Jacobian, Residual,
                                MixedResidual, MixedJacobian, InitialGuess)
from devito.petsc.types.equation import EssentialBC
from devito.petsc.solver_parameters import (linear_solver_parameters,
                                            format_options_prefix)
from devito.petsc.utils import get_funcs, generate_time_mapper


__all__ = ['PETScSolve']


# TODO: Rename this to petsc_solve, petscsolve?
def PETScSolve(target_exprs, target=None, solver_parameters=None, options_prefix=None):
    """
    Returns a symbolic expression representing a linear PETSc solver,
    enriched with all the necessary metadata for execution within an `Operator`.
    When passed to an `Operator`, this symbolic equation triggers code generation
    and lowering to the PETSc backend.

    This function supports both single- and multi-target systems. In the multi-target
    (mixed system) case, the solution vector spans all provided target fields.

    Parameters
    ----------
    target_exprs : Eq or list of Eq, or dict of Function-like -> Eq or list of Eq
        The targets and symbolic expressions defining the system to be solved.

        - Single-field problem:
            Pass a single Eq or list of Eq, and specify `target` separately:
                PETScSolve(Eq1, target)
                PETScSolve([Eq1, Eq2], target)

        - Multi-field (mixed) problem:
            Pass a dictionary mapping each target field to its Eq(s):
                PETScSolve({u: Eq1, v: Eq2})
                PETScSolve({u: [Eq1, Eq2], v: [Eq3, Eq4]})

    target : Function-like
        The function (e.g., `Function`, `TimeFunction`) into which the linear
        system solves. This represents the solution vector updated by the solver.

    solver_parameters : dict, optional
        PETSc solver options.

    Returns
    -------
    Eq:
        A symbolic expression that wraps the linear solver.
        This can be passed directly to a Devito Operator.
    """
    if target is not None:
        return InjectSolve(
            solver_parameters, {target: target_exprs}, options_prefix
        ).build_expr()
    else:
        return InjectMixedSolve(
            solver_parameters, target_exprs, options_prefix
        ).build_expr()


class InjectSolve:
    def __init__(self, solver_parameters=None, target_exprs=None, options_prefix=None):
        self.solver_parameters = linear_solver_parameters(solver_parameters)
        self.time_mapper = None
        self.target_exprs = target_exprs
        self.user_prefix = options_prefix
        self.formatted_prefix = format_options_prefix(options_prefix)

    def build_expr(self):
        target, funcs, field_data = self.linear_solve_args()
        # Placeholder expression for inserting calls to the solver
        linear_solve = LinearSolveExpr(
            funcs,
            self.solver_parameters,
            field_data=field_data,
            time_mapper=self.time_mapper,
            localinfo=localinfo,
            user_prefix=self.user_prefix,
            formatted_prefix=self.formatted_prefix
        )
        return PetscEq(target, linear_solve)

    def linear_solve_args(self):
        target, exprs = next(iter(self.target_exprs.items()))
        exprs = as_tuple(exprs)

        funcs = get_funcs(exprs)
        self.time_mapper = generate_time_mapper(funcs)
        arrays = self.generate_arrays(target)

        exprs = sorted(exprs, key=lambda e: not isinstance(e, EssentialBC))

        jacobian = Jacobian(target, exprs, arrays, self.time_mapper)
        residual = Residual(target, exprs, arrays, self.time_mapper, jacobian.scdiag)
        initial_guess = InitialGuess(target, exprs, arrays)

        field_data = FieldData(
            target=target,
            jacobian=jacobian,
            residual=residual,
            initial_guess=initial_guess,
            arrays=arrays
        )

        return target, tuple(funcs), field_data

    def generate_arrays(self, *targets):
        return {
            t: {
                p: PETScArray(
                    name=f'{p}_{t.name}',
                    target=t,
                    liveness='eager',
                    localinfo=localinfo
                )
                for p in prefixes
            }
            for t in targets
        }


class InjectMixedSolve(InjectSolve):

    def linear_solve_args(self):
        exprs = []
        for e in self.target_exprs.values():
            exprs.extend(e)

        funcs = get_funcs(exprs)
        self.time_mapper = generate_time_mapper(funcs)

        targets = list(self.target_exprs.keys())
        arrays = self.generate_arrays(*targets)

        jacobian = MixedJacobian(
            self.target_exprs, arrays, self.time_mapper
        )

        residual = MixedResidual(
            self.target_exprs, arrays, self.time_mapper,
            jacobian.target_scaler_mapper
        )

        all_data = MultipleFieldData(
            targets=targets,
            arrays=arrays,
            jacobian=jacobian,
            residual=residual
        )

        return targets[0], tuple(funcs), all_data


localinfo = DMDALocalInfo(name='info', liveness='eager')
prefixes = ['y', 'x', 'f', 'b']

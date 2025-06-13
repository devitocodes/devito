from devito.types import Symbol, SteppingDimension
from devito.types.equation import PetscEq
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions
from devito.tools import as_tuple, filter_ordered
from devito.petsc.types import (LinearSolveExpr, PETScArray, DMDALocalInfo,
                                FieldData, MultipleFieldData, Jacobian, Residual,
                                MixedResidual, MixedJacobian, InitialGuess)
from devito.petsc.types.equation import EssentialBC


__all__ = ['PETScSolve']


def PETScSolve(target_exprs, target=None, solver_parameters=None):
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
        return InjectSolve(solver_parameters, {target: target_exprs}).build_expr()
    else:
        return InjectMixedSolve(solver_parameters, target_exprs).build_expr()


class InjectSolve:
    def __init__(self, solver_parameters=None, target_exprs=None):
        self.solver_params = solver_parameters
        self.time_mapper = None
        self.target_exprs = target_exprs

    def build_expr(self):
        target, funcs, fielddata = self.linear_solve_args()

        # Placeholder expression for inserting calls to the solver
        linear_solve = LinearSolveExpr(
            funcs,
            self.solver_params,
            fielddata=fielddata,
            time_mapper=self.time_mapper,
            localinfo=localinfo
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
        initialguess = InitialGuess(target, exprs, arrays)

        field_data = FieldData(
            target=target,
            jacobian=jacobian,
            residual=residual,
            initialguess=initialguess,
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


def generate_time_mapper(funcs):
    """
    Replace time indices with `Symbols` in expressions used within
    PETSc callback functions. These symbols are Uxreplaced at the IET
    level to align with the `TimeDimension` and `ModuloDimension` objects
    present in the initial lowering.
    NOTE: All functions used in PETSc callback functions are attached to
    the `LinearSolveExpr` object, which is passed through the initial lowering
    (and subsequently dropped and replaced with calls to run the solver).
    Therefore, the appropriate time loop will always be correctly generated inside
    the main kernel.
    Examples
    --------
    >>> funcs = [
    >>>     f1(t + dt, x, y),
    >>>     g1(t + dt, x, y),
    >>>     g2(t, x, y),
    >>>     f1(t, x, y)
    >>> ]
    >>> generate_time_mapper(funcs)
    {t + dt: tau0, t: tau1}
    """
    time_indices = list({
        i if isinstance(d, SteppingDimension) else d
        for f in funcs
        for i, d in zip(f.indices, f.dimensions)
        if d.is_Time
    })
    tau_symbs = [Symbol('tau%d' % i) for i in range(len(time_indices))]
    return dict(zip(time_indices, tau_symbs))


def get_funcs(exprs):
    funcs = [
        f for e in exprs
        for f in retrieve_functions(eval_time_derivatives(e.lhs - e.rhs))
    ]
    return filter_ordered(funcs)


localinfo = DMDALocalInfo(name='info', liveness='eager')
prefixes = ['y', 'x', 'f', 'b']

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


def PETScSolve(target_eqns, target=None, solver_parameters=None):
    """
    Returns a symbolic equation representing a linear PETSc solver,
    enriched with all the necessary metadata for execution within an `Operator`.
    When passed to an `Operator`, this symbolic equation triggers code generation
    and lowering to the PETSc backend.

    This function supports both single- and multi-target systems. In the multi-target
    (mixed system) case, the solution vector spans all provided target fields.

    Parameters
    ----------
    target_eqns : Eq or list of Eq, or dict of Function-like -> Eq or list of Eq
        The targets and symbolic equations defining the system to be solved.

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
    Eq
        A symbolic equation that wraps the linear solver.
        This can be passed directly to a Devito Operator.
    """
    if target is not None:
        return InjectSolve(solver_parameters, {target: target_eqns}).build_eq()
    else:
        return InjectMixedSolve(solver_parameters, target_eqns).build_eq()


class InjectSolve:
    def __init__(self, solver_parameters=None, target_eqns=None):
        self.solver_params = solver_parameters
        self.time_mapper = None
        self.target_eqns = target_eqns

    def build_eq(self):
        target, funcs, fielddata = self.linear_solve_args()

        # Placeholder equation for inserting calls to the solver
        linear_solve = LinearSolveExpr(
            funcs,
            self.solver_params,
            fielddata=fielddata,
            time_mapper=self.time_mapper,
            localinfo=localinfo
        )
        return [PetscEq(target, linear_solve)]

    def linear_solve_args(self):
        target, eqns = next(iter(self.target_eqns.items()))
        eqns = as_tuple(eqns)

        funcs = get_funcs(eqns)
        self.time_mapper = generate_time_mapper(funcs)
        arrays = self.generate_arrays_combined(target)

        eqns = sorted(eqns, key=lambda e: not isinstance(e, EssentialBC))

        jacobian = Jacobian(target, eqns, arrays, self.time_mapper)
        residual = Residual(target, eqns, arrays, self.time_mapper, jacobian.scdiag)
        initialguess = InitialGuess(target, eqns, arrays)

        field_data = FieldData(
            target=target,
            jacobian=jacobian,
            residual=residual,
            initialguess=initialguess,
            arrays=arrays
        )

        return target, tuple(funcs), field_data

    def generate_arrays(self, target):
        return {
            p: PETScArray(name=f'{p}_{target.name}',
                          target=target,
                          liveness='eager',
                          localinfo=localinfo)
            for p in prefixes
        }

    def generate_arrays_combined(self, *targets):
        return {target: self.generate_arrays(target) for target in targets}


class InjectMixedSolve(InjectSolve):

    def linear_solve_args(self):
        combined_eqns = []
        for eqns in self.target_eqns.values():
            combined_eqns.extend(eqns)
        funcs = get_funcs(combined_eqns)
        self.time_mapper = generate_time_mapper(funcs)

        coupled_targets = list(self.target_eqns.keys())

        arrays = self.generate_arrays_combined(*coupled_targets)

        jacobian = MixedJacobian(
            self.target_eqns, arrays, self.time_mapper
        )

        residual = MixedResidual(
            self.target_eqns, arrays, self.time_mapper,
            jacobian.target_scaler_mapper
        )

        all_data = MultipleFieldData(
            targets=coupled_targets,
            arrays=arrays,
            jacobian=jacobian,
            residual=residual
        )

        return coupled_targets[0], tuple(funcs), all_data


def generate_time_mapper(funcs):
    """
    Replace time indices with `Symbols` in equations used within
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


def get_funcs(eqns):
    funcs = [
        func
        for eq in eqns
        for func in retrieve_functions(eval_time_derivatives(eq.lhs - eq.rhs))
    ]
    return filter_ordered(funcs)


localinfo = DMDALocalInfo(name='info', liveness='eager')
prefixes = ['y', 'x', 'f', 'b']

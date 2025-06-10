from functools import singledispatch

import sympy
import numpy as np

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.derivative import Derivative
from devito.types import Eq, Symbol, SteppingDimension, TimeFunction
from devito.types.equation import PetscEq
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions
from devito.tools import as_tuple, filter_ordered
from devito.petsc.types import (LinearSolveExpr, PETScArray, DMDALocalInfo,
                                FieldData, MultipleFieldData, SubMatrices)


__all__ = ['PETScSolve', 'EssentialBC']


def PETScSolve(target_eqns, target=None, solver_parameters=None, **kwargs):
    if target is not None:
        return InjectSolve(solver_parameters, {target: target_eqns}).build_eq()
    else:
        return InjectSolveNested(solver_parameters, target_eqns).build_eq()


class InjectSolve:
    def __init__(self, solver_parameters=None, target_eqns=None):
        self.solver_params = solver_parameters
        self.time_mapper = None
        self.target_eqns = target_eqns
        self.cell_area = None

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
        self.cell_area = np.prod(target.grid.spacing_symbols)

        funcs = get_funcs(eqns)
        self.time_mapper = generate_time_mapper(funcs)
        arrays = self.generate_arrays(target)

        return target, tuple(funcs), self.generate_field_data(eqns, target, arrays)

    def generate_field_data(self, eqns, target, arrays):
        # TODO: Apply essential boundary conditions first to preserve operator symmetry 
        # during Jacobian "construction".
        eqns = sorted(eqns, key=lambda e: 0 if isinstance(e, EssentialBC) else 1)

        formfuncs, formrhs = map(
            lambda x: [e for i in x for e in (i if isinstance(i, tuple) else [i])],
            zip(*[self.build_function_eqns(eq, target, arrays) for eq in eqns])
        )
        matvecs = [e for i in [self.build_matvec_eqns(eq, target, arrays) for eq in eqns]
                   for e in (i if isinstance(i, (tuple)) else [i])]
        from IPython import embed; embed()
        initialguess = [
            eq for eq in
            (self.make_initial_guess(e, target, arrays) for e in eqns)
            if eq is not None
        ]

        return FieldData(
            target=target,
            matvecs=matvecs,
            formfuncs=formfuncs,
            formrhs=formrhs,
            initialguess=initialguess,
            arrays=arrays
        )

    def build_function_eqns(self, eq, target, arrays):
        b, F_target, _, targets = separate_eqn(eq, target)
        formfunc = self.make_formfunc(eq, F_target, arrays, targets)
        formrhs = self.make_rhs(eq, b, arrays)

        return (formfunc, formrhs)

    def build_matvec_eqns(self, eq, target, arrays):
        b, F_target, _, targets = separate_eqn(eq, target)
        if not F_target:
            return None
        matvec = self.make_matvec(eq, F_target, arrays, targets)
        return matvec

    def make_matvec(self, eq, F_target, arrays, targets):
        if isinstance(eq, EssentialBC):
            # NOTE: Until PetscSection + DMDA is supported, we leave
            # the essential BCs in the solver.
            # Trivial equations for bc rows -> place 1.0 on diagonal (scaled)
            # and zero symmetrically.
            rhs = arrays['x']
            zero_column = Eq(arrays['x'], 0.0, subdomain=eq.subdomain)
            return (EssentialBC(arrays['y'], rhs, subdomain=eq.subdomain), zero_column)
        else:
            rhs = F_target.subs(targets_to_arrays(arrays['x'], targets))
            rhs = rhs.subs(self.time_mapper)
            rhs = rhs * self.cell_area
        return Eq(arrays['y'], rhs, subdomain=eq.subdomain)

    def make_formfunc(self, eq, F_target, arrays, targets):
        if isinstance(eq, EssentialBC):
            # The initial guess already satisfies the essential boundary conditions,
            # so this term will always be zero. It's included here to allow
            # testing of the Jacobian via finite differences.
            rhs = arrays['x'] - eq.rhs
            # Create equation to handle essential boundary conditions - we
            # move the essential BCs to the right-hand side
            # and zero the corresponding column in the Jacobian.
            # TODO: extend this to mixed problems
            move_bc = Eq(arrays['x'], eq.rhs, subdomain=eq.subdomain)
            return (EssentialBC(arrays['f'], rhs, subdomain=eq.subdomain), move_bc)
        else:
            if isinstance(F_target, (int, float)):
                rhs = F_target * self.cell_area
            else:
                rhs = F_target.subs(targets_to_arrays(arrays['x'], targets))
                rhs = rhs.subs(self.time_mapper)
                rhs = rhs * self.cell_area
        return Eq(arrays['f'], rhs, subdomain=eq.subdomain)

    def make_rhs(self, eq, b, arrays):
        rhs = 0. if isinstance(eq, EssentialBC) else b.subs(self.time_mapper)
        rhs = rhs*self.cell_area
        return Eq(arrays['b'], rhs, subdomain=eq.subdomain)

    def make_initial_guess(self, eq, target, arrays):
        """
        Enforce initial guess to satisfy essential BCs.
        # TODO: For time-stepping, only enforce these once outside the time loop
        and use the previous time-step solution as the initial guess for next time step.
        # TODO: Extend this to "coupled".
        """
        if isinstance(eq, EssentialBC):
            assert eq.lhs == target
            return Eq(
                arrays['x'], eq.rhs,
                subdomain=eq.subdomain
            )
        else:
            return None

    def generate_arrays(self, target):
        return {
            p: PETScArray(name=f'{p}_{target.name}',
                          target=target,
                          liveness='eager',
                          localinfo=localinfo)
            for p in prefixes
        }


class InjectSolveNested(InjectSolve):
    def linear_solve_args(self):
        combined_eqns = []
        for eqns in self.target_eqns.values():
            combined_eqns.extend(eqns)
        funcs = get_funcs(combined_eqns)
        self.time_mapper = generate_time_mapper(funcs)

        coupled_targets = list(self.target_eqns.keys())
        jacobian = SubMatrices(coupled_targets)

        arrays = self.generate_arrays_combined(*coupled_targets)

        all_data = MultipleFieldData(submatrices=jacobian, arrays=arrays,
                                     targets=coupled_targets)

        self.cell_area = np.prod(all_data.grid.spacing_symbols)

        for target, eqns in self.target_eqns.items():
            eqns = as_tuple(eqns)
            self.update_jacobian(eqns, target, jacobian, arrays[target])

            formfuncs = [
                self.build_function_eqns(eq, target, coupled_targets, arrays)
                for eq in eqns
            ]
            all_data.extend_formfuncs(formfuncs)

        return target, tuple(funcs), all_data

    def update_jacobian(self, eqns, target, jacobian, arrays):
        for submat, mtvs in jacobian.submatrices[target].items():
            # TODO: maintain symmetry for coupled
            matvecs = [e for i in [self.build_matvec_eqns(eq, mtvs['derivative_wrt'], arrays)
                       for eq in eqns] for e in (i if isinstance(i, (tuple)) else [i])]
            matvecs = [m for m in matvecs if m is not None]

            if matvecs:
                jacobian.set_submatrix(target, submat, matvecs)

    def build_function_eqns(self, eq, main_target, coupled_targets, arrays):
        zeroed = eq.lhs - eq.rhs

        # TODO: clean up, test coupled with time dependence
        zeroed_eqn = Eq(zeroed, 0)
        zeroed_eqn = eval_time_derivatives(zeroed)

        mapper = {}
        for t in coupled_targets:
            target_funcs = generate_targets(Eq(zeroed, 0), t)
            mapper.update(targets_to_arrays(arrays[t]['x'], target_funcs))

        if isinstance(eq, EssentialBC):
            rhs = arrays[main_target]['x'] - eq.rhs
        else:
            if isinstance(zeroed, (int, float)):
                rhs = zeroed * self.cell_area
            else:
                rhs = zeroed.subs(mapper)
                rhs = rhs.subs(self.time_mapper)
                rhs = rhs * self.cell_area

        return Eq(arrays[main_target]['f'], rhs, subdomain=eq.subdomain)

    def generate_arrays_combined(self, *targets):
        return {
            target: {
                p: PETScArray(
                    name=f'{p}_{target.name}',
                    target=target,
                    liveness='eager',
                    localinfo=localinfo
                )
                for p in prefixes
            }
            for target in targets
        }

    def make_matvec(self, eq, F_target, arrays, targets):
        if isinstance(eq, EssentialBC):
            # TODO: SHOULDN'T NEED this subclass once fixed mixed problems + symmetry
            # NOTE: Until PetscSection + DMDA is supported, we leave
            # the essential BCs in the solver.
            # Trivial equations for bc rows -> place 1.0 on diagonal
            # and zero symmetrically. FIX FOR MIXED PROBLEMS
            rhs = arrays['x']
        else:
            rhs = F_target.subs(targets_to_arrays(arrays['x'], targets))
            rhs = rhs.subs(self.time_mapper)
            rhs = rhs * self.cell_area
        return Eq(arrays['y'], rhs, subdomain=eq.subdomain)


class EssentialBC(Eq):
    """
    A special equation representing an essential boundary condition.
    It is used to handle essential boundary conditions in the PETSc solver.
    Until PetscSection + DMDA is supported, we treat essential boundary conditions
    as trivial equations in the solver, where we place 1.0 on the diagonal of the jacobian, 
    zero symmetrically and move the essential boundary condition to the right-hand side.

    NOTE: When users define essential boundary conditions, they need to ensure that
    the SubDomains do not overlap. Solver will still run but may see unexpected behaviour
    at boundaries.
    """
    pass


def separate_eqn(eqn, target):
    """
    Separate the equation into two separate expressions,
    where F(target) = b.
    """
    zeroed_eqn = Eq(eqn.lhs - eqn.rhs, 0)
    zeroed_eqn = eval_time_derivatives(zeroed_eqn.lhs)
    target_funcs = set(generate_targets(zeroed_eqn, target))
    b, F_target = remove_targets(zeroed_eqn, target_funcs)
    return -b, F_target, zeroed_eqn, target_funcs


def generate_targets(eq, target):
    """
    Extract all the functions that share the same time index as the target
    but may have different spatial indices.
    """
    funcs = retrieve_functions(eq)
    if isinstance(target, TimeFunction):
        time_idx = target.indices[target.time_dim]
        targets = [
            f for f in funcs if f.function is target.function and time_idx
            in f.indices
        ]
    else:
        targets = [f for f in funcs if f.function is target.function]
    return targets


def targets_to_arrays(array, targets):
    """
    Map each target in `targets` to a corresponding array generated from `array`,
    matching the spatial indices of the target.
    Example:
    --------
    >>> array
    vec_u(x, y)
    >>> targets
    {u(t + dt, x + h_x, y), u(t + dt, x - h_x, y), u(t + dt, x, y)}
    >>> targets_to_arrays(array, targets)
    {u(t + dt, x - h_x, y): vec_u(x - h_x, y),
     u(t + dt, x + h_x, y): vec_u(x + h_x, y),
     u(t + dt, x, y): vec_u(x, y)}
    """
    space_indices = [
        tuple(f.indices[d] for d in f.space_dimensions) for f in targets
    ]
    array_targets = [
        array.subs(dict(zip(array.indices, i))) for i in space_indices
    ]
    return dict(zip(targets, array_targets))


@singledispatch
def remove_targets(expr, targets):
    return (0, expr) if expr in targets else (expr, 0)


@remove_targets.register(sympy.Add)
def _(expr, targets):
    if not any(expr.has(t) for t in targets):
        return (expr, 0)

    args_b, args_F = zip(*(remove_targets(a, targets) for a in expr.args))
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_targets.register(Mul)
def _(expr, targets):
    if not any(expr.has(t) for t in targets):
        return (expr, 0)

    args_b, args_F = zip(*[remove_targets(a, targets) if any(a.has(t) for t in targets)
                           else (a, a) for a in expr.args])
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_targets.register(Derivative)
def _(expr, targets):
    return (0, expr) if any(expr.has(t) for t in targets) else (expr, 0)


@singledispatch
def centre_stencil(expr, target):
    """
    Extract the centre stencil from an expression. Its coefficient is what
    would appear on the diagonal of the matrix system if the matrix were
    formed explicitly.
    """
    return expr if expr == target else 0


@centre_stencil.register(sympy.Add)
def _(expr, target):
    if not expr.has(target):
        return 0

    args = [centre_stencil(a, target) for a in expr.args]
    return expr.func(*args, evaluate=False)


@centre_stencil.register(Mul)
def _(expr, target):
    if not expr.has(target):
        return 0

    args = []
    for a in expr.args:
        if not a.has(target):
            args.append(a)
        else:
            args.append(centre_stencil(a, target))

    return expr.func(*args, evaluate=False)


@centre_stencil.register(Derivative)
def _(expr, target):
    if not expr.has(target):
        return 0
    args = [centre_stencil(a, target) for a in expr.evaluate.args]
    return expr.evaluate.func(*args)


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

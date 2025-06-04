from functools import singledispatch

import sympy
import numpy as np
from itertools import chain
from collections import defaultdict

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.derivative import Derivative
from devito.types import Eq, Symbol, SteppingDimension, TimeFunction
from devito.types.equation import PetscEq
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions
from devito.symbolics.extraction import (separate_eqn, centre_stencil,
                                         generate_targets)
from devito.tools import as_tuple, filter_ordered
from devito.petsc.types import (LinearSolveExpr, PETScArray, DMDALocalInfo,
                                FieldData, MultipleFieldData, Jacobian, Residual,
                                MixedResidual, MixedJacobian)
from devito.petsc.types.equation import EssentialBC, ZeroRow, ZeroColumn


__all__ = ['PETScSolve']


def PETScSolve(target_eqns, target=None, solver_parameters=None, **kwargs):
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
        arrays = self.generate_arrays(target)

        eqns = sorted(eqns, key=lambda e: 0 if isinstance(e, EssentialBC) else 1)

        jacobian = Jacobian(target, self.time_mapper, arrays)
        jacobian.build_block(eqns)

        scale = 1.0

        residual = Residual(target, self.time_mapper, arrays, scale)
        residual.build_equations(eqns)

        initialguess = [
            eq for eq in
            (self.make_initial_guess(e, target, arrays) for e in eqns)
            if eq is not None
        ]

        field_data = FieldData(
            target=target,
            jacobian=jacobian,
            residual=residual,
            initialguess=initialguess,
            arrays=arrays
        )

        return target, tuple(funcs), field_data

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


class InjectMixedSolve(InjectSolve):

    def linear_solve_args(self):

        combined_eqns = []
        for eqns in self.target_eqns.values():
            combined_eqns.extend(eqns)
        funcs = get_funcs(combined_eqns)
        self.time_mapper = generate_time_mapper(funcs)

        coupled_targets = list(self.target_eqns.keys())

        arrays = self.generate_arrays_combined(*coupled_targets)

        jacobian = MixedJacobian(coupled_targets, self.time_mapper, arrays)
        jacobian.build_blocks(self.target_eqns)

        residual = MixedResidual(coupled_targets, self.time_mapper, arrays)
        residual.build_equations(self.target_eqns)

        all_data = MultipleFieldData(
            targets=coupled_targets,
            arrays=arrays,
            jacobian=jacobian,
            residual=residual
        )

        # TODO: rethink what target to return here???
        return coupled_targets[0], tuple(funcs), all_data

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

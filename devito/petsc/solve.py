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
                                FieldData, MultipleFieldData, Jacobian)


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
        # TODO: make this _
        self.cell_area = None
        # self._centre_stencils = set()
        self._diag_scale = defaultdict(set)

    @property
    def diag_scale(self):
        return self._diag_scale

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
        # Apply essential boundary conditions first to preserve
        # operator symmetry during Jacobian "construction"
        eqns = sorted(eqns, key=lambda e: 0 if isinstance(e, EssentialBC) else 1)

        matvecs = [e for eq in eqns for e in self.build_matvec_eq(eq, target, arrays)]

        formfuncs, formrhs = map(
            lambda x: [e for i in x for e in i],
            zip(*[self.build_function_eq(eq, target, arrays) for eq in eqns])
        )

        stencils = set()
        for eq in matvecs:
            if not isinstance(eq, EssentialBC):
                stencil = centre_stencil(eq.rhs, arrays['x'], as_coeff=True)
                stencils.add(stencil)

        if len(stencils) > 1:
            # Scaling of jacobian is therefore ambiguous, potentially could average across the subblock
            # for now just set to trivial 1.0
            scale = 1.0
        else:
            scale = next(iter(stencils))

        # from IPython import embed; embed()
        matvecs = self.scale_essential_bcs(matvecs, scale)
        formfuncs = self.scale_essential_bcs(formfuncs, scale)

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

    def build_function_eq(self, eq, target, arrays):
        b, F_target, _, targets = separate_eqn(eq, target)
        formfunc = self.make_formfunc(eq, F_target, arrays, targets)
        formrhs = self.make_rhs(eq, b, arrays)

        return (formfunc, formrhs)

    def build_matvec_eq(self, eq, target, arrays):
        b, F_target, _, targets = separate_eqn(eq, target)
        if F_target:
            return self.make_matvec(eq, F_target, arrays, targets)
        return (None,)

    def make_matvec(self, eq, F_target, arrays, targets):
        if isinstance(eq, EssentialBC):
            # NOTE: Until PetscSection + DMDA is supported, we leave
            # the essential BCs in the solver.
            # Trivial equations for bc rows -> place 1.0 on diagonal (scaled)
            # and zero symmetrically.
            rhs = arrays['x']
            zero_row = ZeroRow(arrays['y'], rhs, subdomain=eq.subdomain)
            zero_column = ZeroColumn(arrays['x'], 0.0, subdomain=eq.subdomain)
            return (zero_row, zero_column)
        else:
            rhs = F_target.subs(targets_to_arrays(arrays['x'], targets))
            rhs = rhs.subs(self.time_mapper) * self.cell_area
            # TODO: Average centre stencils if they vary, to scale essential BC rows.
            # self.centre = centre_stencil(rhs, arrays['x'], as_coeff=True)
            # stencil = centre_stencil(rhs, arrays['x'], as_coeff=True)
            # self._centre_stencils[arrays['x']].add(stencil)
            # self._centre_stencils.add(stencil)

        return as_tuple(Eq(arrays['y'], rhs, subdomain=eq.subdomain))

    def make_formfunc(self, eq, F_target, arrays, targets):
        if isinstance(eq, EssentialBC):
            # The initial guess satisfies the essential BCs, so this term is zero.
            # Still included to support Jacobian testing via finite differences.
            rhs = arrays['x'] - eq.rhs
            zero_row = ZeroRow(arrays['f'], rhs, subdomain=eq.subdomain)
            # Move essential boundary condition to the right-hand side
            zero_col = ZeroColumn(arrays['x'], eq.rhs, subdomain=eq.subdomain)
            return (zero_row, zero_col)
        else:
            if isinstance(F_target, (int, float)):
                rhs = F_target * self.cell_area
            else:
                rhs = F_target.subs(targets_to_arrays(arrays['x'], targets))
                rhs = rhs.subs(self.time_mapper) * self.cell_area
        return as_tuple(Eq(arrays['f'], rhs, subdomain=eq.subdomain))

    def make_rhs(self, eq, b, arrays):
        rhs = 0. if isinstance(eq, EssentialBC) else b.subs(self.time_mapper)
        rhs = rhs * self.cell_area
        return as_tuple(Eq(arrays['b'], rhs, subdomain=eq.subdomain))

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

    def scale_essential_bcs(self, equations, scale):
        """
        Scale the essential boundary rows so that the Jacobian has a constant diagonal,
        thereby reducing its condition number.
        """
        # # stencils = self.centre_stencils[arrays['x']]
        # if len(stencils) > 1:
        #     # Scaling of jacobian is therefore ambiguous, potentially could averge across the subblock
        #     # for now just set to trivial 1.0
        #     scale = 1.0
        # else:
        #     scale = next(iter(stencils))
        return [
            eq._rebuild(rhs=scale * eq.rhs) if isinstance(eq, ZeroRow) else eq
            for eq in equations
        ]


class InjectSolveNested(InjectSolve):

    def linear_solve_args(self):

        combined_eqns = []
        for eqns in self.target_eqns.values():
            combined_eqns.extend(eqns)
        funcs = get_funcs(combined_eqns)
        self.time_mapper = generate_time_mapper(funcs)

        coupled_targets = list(self.target_eqns.keys())
        jacobian = Jacobian(coupled_targets)

        arrays = self.generate_arrays_combined(*coupled_targets)

        all_data = MultipleFieldData(jacobian=jacobian, arrays=arrays,
                                     targets=coupled_targets)

        self.cell_area = np.prod(all_data.grid.spacing_symbols)

        all_formfuncs = []

        for target, eqns in self.target_eqns.items():

            # Update all rows of the Jacobian for this target
            self.update_jacobian(as_tuple(eqns), target, jacobian, arrays[target])

            formfuncs = chain.from_iterable(
                self.build_function_eq(eq, target, coupled_targets, arrays)
                for eq in as_tuple(eqns)
            )
            # from IPython import embed; embed()
            scale, = self._diag_scale[arrays[target]['x']]
            all_formfuncs.extend(self.scale_essential_bcs(formfuncs, scale))

        formfuncs = tuple(sorted(
            all_formfuncs, key=lambda e: not isinstance(e, EssentialBC)
        ))
        all_data.extend_formfuncs(formfuncs)

        return target, tuple(funcs), all_data

    def update_jacobian(self, eqns, target, jacobian, arrays):

        for submat, mtvs in jacobian.submatrices[target].items():
            matvecs = [
                e for eq in eqns for e in
                self.build_matvec_eq(eq, mtvs['derivative_wrt'], arrays)
            ]
            matvecs = [m for m in matvecs if m is not None]

            if submat in jacobian.diagonal_submatrix_keys:
                stencils = set()
                for eq in matvecs:
                    if not isinstance(eq, EssentialBC):
                        stencil = centre_stencil(eq.rhs, arrays['x'], as_coeff=True)
                        stencils.add(stencil)
                # from IPython import embed; embed()
                if len(stencils) > 1:
                    # Scaling of jacobian is therefore ambiguous, potentially could average across the subblock
                    # for now just set to trivial 1.0
                    # TODO: doens't need to be a defaultdict, just a dict?
                    self._diag_scale[arrays['x']].add(1.0)
                    scale = 1.0
                else:
                    scale = next(iter(stencils))
                    self._diag_scale[arrays['x']].add(scale)
                    # scale = next(iter(stencils))

                matvecs = self.scale_essential_bcs(matvecs, scale)

            matvecs = tuple(sorted(matvecs, key=lambda e: not isinstance(e, EssentialBC)))

            if matvecs:
                jacobian.set_submatrix(target, submat, matvecs)

    def build_function_eq(self, eq, main_target, coupled_targets, arrays):
        zeroed = eq.lhs - eq.rhs

        zeroed_eqn = Eq(eq.lhs - eq.rhs, 0)
        eval_zeroed_eqn = eval_time_derivatives(zeroed_eqn.lhs)

        mapper = {}
        for t in coupled_targets:
            target_funcs = set(generate_targets(Eq(eval_zeroed_eqn, 0), t))
            mapper.update(targets_to_arrays(arrays[t]['x'], target_funcs))

        if isinstance(eq, EssentialBC):
            rhs = arrays[main_target]['x'] - eq.rhs
            zero_row = ZeroRow(
                arrays[main_target]['f'], rhs, subdomain=eq.subdomain
            )
            zero_col = ZeroColumn(
                arrays[main_target]['x'], eq.rhs, subdomain=eq.subdomain
            )
            return (zero_row, zero_col)
        else:
            if isinstance(zeroed, (int, float)):
                rhs = zeroed * self.cell_area
            else:
                rhs = zeroed.subs(mapper)
                rhs = rhs.subs(self.time_mapper)*self.cell_area

        return as_tuple(Eq(arrays[main_target]['f'], rhs, subdomain=eq.subdomain))

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


class EssentialBC(Eq):
    """
    A special equation used to handle essential boundary conditions
    in the PETSc solver. Until PetscSection + DMDA is supported,
    we treat essential boundary conditions as trivial equations
    in the solver, where we place 1.0 (scaled) on the diagonal of
    the jacobian, zero symmetrically and move the boundary
    data to the right-hand side.

    NOTE: When users define essential boundary conditions, they need to ensure that
    the SubDomains do not overlap. Solver will still run but may see unexpected behaviour
    at boundaries. This will be documented in the PETSc examples.
    """
    pass


class ZeroRow(EssentialBC):
    """
    Equation used to zero the row of the Jacobian corresponding
    to an essential BC.
    This is only used interally by the compiler, not by users.
    """
    pass


class ZeroColumn(EssentialBC):
    """
    Equation used to zero the column of the Jacobian corresponding
    to an essential BC.
    This is only used interally by the compiler, not by users.
    """
    pass


# def separate_eqn(eqn, target):
#     """
#     Separate the equation into two separate expressions,
#     where F(target) = b.
#     """
#     zeroed_eqn = Eq(eqn.lhs - eqn.rhs, 0)
#     zeroed_eqn = eval_time_derivatives(zeroed_eqn.lhs)
#     target_funcs = set(generate_targets(zeroed_eqn, target))
#     b, F_target = remove_targets(zeroed_eqn, target_funcs)
#     return -b, F_target, zeroed_eqn, target_funcs


# def generate_targets(eq, target):
#     """
#     Extract all the functions that share the same time index as the target
#     but may have different spatial indices.
#     """
#     funcs = retrieve_functions(eq)
#     if isinstance(target, TimeFunction):
#         time_idx = target.indices[target.time_dim]
#         targets = [
#             f for f in funcs if f.function is target.function and time_idx
#             in f.indices
#         ]
#     else:
#         targets = [f for f in funcs if f.function is target.function]
#     return targets


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

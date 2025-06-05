import sympy

from itertools import chain

from devito.tools import Reconstructable, sympy_mutex, as_tuple
from devito.tools.dtypes_lowering import dtype_mapper
from devito.petsc.utils import petsc_variables
from devito.symbolics.extraction import separate_eqn, generate_targets, centre_stencil
from devito.petsc.types.equation import EssentialBC, ZeroRow, ZeroColumn
from devito.types.equation import Eq
from devito.operations.solve import eval_time_derivatives


class MetaData(sympy.Function, Reconstructable):
    def __new__(cls, expr, **kwargs):
        with sympy_mutex:
            obj = sympy.Function.__new__(cls, expr)
        obj._expr = expr
        return obj

    @property
    def expr(self):
        return self._expr


class Initialize(MetaData):
    pass


class Finalize(MetaData):
    pass


class LinearSolveExpr(MetaData):
    """
    A symbolic expression passed through the Operator, containing the metadata
    needed to execute a linear solver. Linear problems are handled with
    `SNESSetType(snes, KSPONLY)`, enabling a unified interface for both
    linear and nonlinear solvers.
    # TODO: extend this
    defaults:
        - 'ksp_type': String with the name of the PETSc Krylov method.
           Default is 'gmres' (Generalized Minimal Residual Method).
           https://petsc.org/main/manualpages/KSP/KSPType/
        - 'pc_type': String with the name of the PETSc preconditioner.
           Default is 'jacobi' (i.e diagonal scaling preconditioning).
           https://petsc.org/main/manualpages/PC/PCType/
        KSP tolerances:
        https://petsc.org/release/manualpages/KSP/KSPSetTolerances/
        - 'ksp_rtol': Relative convergence tolerance. Default
                      is 1e-5.
        - 'ksp_atol': Absolute convergence for tolerance. Default
                      is 1e-50.
        - 'ksp_divtol': Divergence tolerance, amount residual norm can
                        increase before `KSPConvergedDefault()` concludes
                        that the method is diverging. Default is 1e5.
        - 'ksp_max_it': Maximum number of iterations to use. Default
                        is 1e4.
    """

    __rargs__ = ('expr',)
    __rkwargs__ = ('solver_parameters', 'fielddata', 'time_mapper',
                   'localinfo')

    defaults = {
        'ksp_type': 'gmres',
        'pc_type': 'jacobi',
        'ksp_rtol': 1e-5,  # Relative tolerance
        'ksp_atol': 1e-50,  # Absolute tolerance
        'ksp_divtol': 1e5,  # Divergence tolerance
        'ksp_max_it': 1e4  # Maximum iterations
    }

    def __new__(cls, expr, solver_parameters=None,
                fielddata=None, time_mapper=None, localinfo=None, **kwargs):

        if solver_parameters is None:
            solver_parameters = cls.defaults
        else:
            for key, val in cls.defaults.items():
                solver_parameters[key] = solver_parameters.get(key, val)

        with sympy_mutex:
            obj = sympy.Function.__new__(cls, expr)

        obj._expr = expr
        obj._solver_parameters = solver_parameters
        obj._fielddata = fielddata if fielddata else FieldData()
        obj._time_mapper = time_mapper
        obj._localinfo = localinfo
        return obj

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.expr)

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    def __hash__(self):
        return hash(self.expr)

    def __eq__(self, other):
        return (isinstance(other, LinearSolveExpr) and
                self.expr == other.expr)

    @property
    def expr(self):
        return self._expr

    @property
    def fielddata(self):
        return self._fielddata

    @property
    def solver_parameters(self):
        return self._solver_parameters

    @property
    def time_mapper(self):
        return self._time_mapper

    @property
    def localinfo(self):
        return self._localinfo

    @property
    def grid(self):
        return self.fielddata.grid

    @classmethod
    def eval(cls, *args):
        return None

    func = Reconstructable._rebuild


class FieldData:
    def __init__(self, target=None, jacobian=None, residual=None,
                 initialguess=None, arrays=None, **kwargs):
        self._target = target
        petsc_precision = dtype_mapper[petsc_variables['PETSC_PRECISION']]
        if self._target.dtype != petsc_precision:
            raise TypeError(
                f"Your target dtype must match the precision of your "
                f"PETSc configuration. "
                f"Expected {petsc_precision}, but got {self._target.dtype}."
            )
        self._jacobian = jacobian
        self._residual = residual
        self._initialguess = initialguess
        self._arrays = arrays

    @property
    def target(self):
        return self._target

    @property
    def jacobian(self):
        return self._jacobian

    @property
    def residual(self):
        return self._residual

    @property
    def initialguess(self):
        return self._initialguess

    @property
    def arrays(self):
        return self._arrays

    @property
    def space_dimensions(self):
        return self.target.space_dimensions

    @property
    def grid(self):
        return self.target.grid

    @property
    def space_order(self):
        return self.target.space_order

    @property
    def targets(self):
        return as_tuple(self.target)


class MultipleFieldData(FieldData):
    def __init__(self, targets, arrays, jacobian=None, residual=None):
        self._targets = as_tuple(targets)
        self._arrays = arrays
        self._jacobian = jacobian
        self._residual = residual

    @property
    def space_dimensions(self):
        space_dims = {t.space_dimensions for t in self.targets}
        if len(space_dims) > 1:
            # TODO: This may not actually have to be the case, but enforcing it for now
            raise ValueError(
                "All targets within a PETScSolve have to have the same space dimensions."
            )
        return space_dims.pop()

    @property
    def grid(self):
        grids = [t.grid for t in self.targets]
        if len(set(grids)) > 1:
            raise ValueError(
                "All targets within a PETScSolve have to have the same grid."
            )
        return grids.pop()

    @property
    def space_order(self):
        # NOTE: since we use DMDA to create vecs for the coupled solves,
        # all fields must have the same space order
        # ... re think this? limitation. For now, just force the
        # space order to be the same.
        # This isn't a problem for segregated solves.
        space_orders = [t.space_order for t in self.targets]
        if len(set(space_orders)) > 1:
            raise ValueError(
                "All targets within a PETScSolve have to have the same space order."
            )
        return space_orders.pop()

    @property
    def targets(self):
        return self._targets


class Jacobian:
    """
    Represents a Jacobian matrix in a matrix-free form.

    This Jacobian is defined implicitly via matrix-vector products
    derived from the symbolic equations provided in `matvecs`.

    It assumes the problem is linear, meaning the Jacobian
    corresponds to a constant coefficient matrix and does not
    require explicit symbolic differentiation.
    """
    def __init__(self, target, eqns, arrays, time_mapper):
        self.target = target
        self.eqns = eqns
        self.arrays = arrays
        self.time_mapper = time_mapper
        self._build_matvecs()

    @property
    def matvecs(self):
        return self._matvecs

    @property
    def scdiag(self):
        return self._scdiag

    def _build_matvecs(self):
        matvecs = [
            e for eq in self.eqns for e in
            self._build_matvec_eq(eq)
            if e is not None
        ]
        matvecs = tuple(sorted(matvecs, key=lambda e: not isinstance(e, EssentialBC)))

        matvecs = self._scale_non_bcs(matvecs)
        scdiag = self._compute_scdiag(matvecs)
        matvecs = self._scale_bcs(matvecs, scdiag)

        self._matvecs = matvecs
        self._scdiag = scdiag

    def _build_matvec_eq(self, eq, target=None, arrays=None):
        target = target or self.target
        arrays = arrays or self.arrays

        b, F_target, _, targets = separate_eqn(eq, target)
        if F_target:
            return self._make_matvec(eq, F_target, targets, arrays)
        return (None,)

    def _make_matvec(self, eq, F_target, targets, arrays):
        if isinstance(eq, EssentialBC):
            # NOTE: Essential BCs are trivial equations in the solver.
            # See `EssentialBC` for more details.
            rhs = arrays['x']
            zero_row = ZeroRow(arrays['y'], rhs, subdomain=eq.subdomain)
            zero_column = ZeroColumn(arrays['x'], 0., subdomain=eq.subdomain)
            return (zero_row, zero_column)
        else:
            rhs = F_target.subs(targets_to_arrays(arrays['x'], targets))
            rhs = rhs.subs(self.time_mapper)
        return as_tuple(Eq(arrays['y'], rhs, subdomain=eq.subdomain))

    def _scale_non_bcs(self, matvecs, target=None):
        target = target or self.target
        vol = target.grid.symbolic_volume_cell

        return [
            m if isinstance(m, EssentialBC) else m._rebuild(rhs=m.rhs * vol)
            for m in matvecs
        ]

    def _compute_scdiag(self, matvecs, arrays=None):
        """
        """
        arrays = arrays or self.arrays

        centres = {
            centre_stencil(m.rhs, arrays['x'], as_coeff=True)
            for m in matvecs if not isinstance(m, EssentialBC)
        }
        # add comments
        return centres.pop() if len(centres) == 1 else 1.0

    def _scale_bcs(self, matvecs, scdiag):
        """
        Scale the essential BCs
        """
        return [
            m._rebuild(rhs=m.rhs * scdiag) if isinstance(m, ZeroRow) else m
            for m in matvecs
        ]


class SubMatrixBlock:
    def __init__(self, name, matvecs, scdiag,
                 row_target, col_target, row_idx, col_idx, linear_idx):
        self.name = name
        self.matvecs = matvecs
        self.scdiag = scdiag
        self.row_target = row_target
        self.col_target = col_target
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.linear_idx = linear_idx

    def is_diag(self):
        return self.row_idx == self.col_idx

    def __repr__(self):
        return (f"<SubMatrixBlock {self.name}>")


class MixedJacobian(Jacobian):
    """
    Represents a Jacobian for a linear system with a solution vector
    composed of multiple fields (targets).

    Defines matrix-vector products for each sub-block,
    each with its own generated callback. The matrix may be treated
    as monolithic or block-structured in PETSc, but sub-block
    callbacks are generated in both cases.

    Assumes a **linear** problem, so this Jacobian corresponds to a
    coefficient matrix and does not require differentiation.

    # TODO: pcfieldsplit support for each block
    """
    def __init__(self, target_eqns, arrays, time_mapper):
        """
        """
        self.targets = as_tuple(target_eqns.keys())
        self.arrays = arrays
        self.time_mapper = time_mapper
        self._submatrices = []
        self._build_blocks(target_eqns)

    @property
    def submatrices(self):
        """
        Return a list of all submatrix blocks.
        Each block contains metadata about the matrix-vector products.
        """
        return self._submatrices

    @property
    def no_submatrices(self):
        """
        Return the number of submatrix blocks.
        """
        return len(self._submatrices)

    @property
    def nonzero_submatrices(self):
        """Return SubMatrixBlock objects that have non-empty matvecs."""
        return [submat for submat in self.submatrices if submat.matvecs]

    @property
    def target_scaler_mapper(self):
        """
        Map each row_target index to its diagonal submatrix's scaler.
        """
        mapper = {}
        for sm in self.submatrices:
            if sm.row_idx == sm.col_idx:
                mapper[sm.row_target] = sm.scdiag
        return mapper

    def _build_blocks(self, target_eqns):
        """
        Build all SubMatrixBlock objects for the Jacobian.
        """
        for i, row_target in enumerate(self.targets):
            eqns = target_eqns[row_target]
            arrays = self.arrays[row_target]
            for j, col_target in enumerate(self.targets):
                matvecs = [
                    e for eq in eqns for e in
                    self._build_matvec_eq(eq, col_target, arrays)
                ]
                matvecs = [m for m in matvecs if m is not None]
                # Sort to put EssentialBC first if any
                matvecs = tuple(
                    sorted(matvecs, key=lambda e: not isinstance(e, EssentialBC))
                )
                matvecs = self._scale_non_bcs(matvecs, row_target)
                scdiag = self._compute_scdiag(matvecs, arrays)
                matvecs = self._scale_bcs(matvecs, scdiag)

                name = f'J{i}{j}'
                block = SubMatrixBlock(
                    name=name,
                    matvecs=matvecs,
                    scdiag=scdiag,
                    row_target=row_target,
                    col_target=col_target,
                    row_idx=i,
                    col_idx=j,
                    linear_idx=i * len(self.targets) + j
                )
                self._submatrices.append(block)

    def get_submatrix(self, row_idx, col_idx):
        """
        Return the SubMatrixBlock at (row_idx, col_idx), or None if not found.
        """
        for sm in self.submatrices:
            if sm.row_idx == row_idx and sm.col_idx == col_idx:
                return sm
        return None

    def __repr__(self):
        summary = ', '.join(
            f"{sm.name} (row={sm.row_idx}, col={sm.col_idx})"
            for sm in self.submatrices
        )
        return f"<MixedJacobian with {self.no_submatrices} submatrices: [{summary}]>"


class Residual:
    """
    """
    def __init__(self, target, eqns, arrays, time_mapper, scdiag):
        self.target = target
        self.eqns = eqns
        self.arrays = arrays
        self.time_mapper = time_mapper
        self.scdiag = scdiag
        self._build_equations()

    @property
    def formfuncs(self):
        """
        """
        return self._formfuncs

    @property
    def formrhs(self):
        """
        """
        return self._formrhs

    def _build_equations(self):
        """
        """
        funcs = []
        rhs = []

        for eq in self.eqns:
            b, F_target, _, targets = separate_eqn(eq, self.target)
            funcs.extend(self._make_F_target(eq, F_target, targets))
            # TODO: If b is zero then don't need a rhs vector+callback
            rhs.extend(self._make_b(eq, b))

        self._formfuncs = [self._scale_bcs(eq) for eq in funcs]
        self._formrhs = rhs

    def _make_F_target(self, eq, F_target, targets):
        arrays = self.arrays
        volume = self.target.grid.symbolic_volume_cell
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
                rhs = F_target * volume
            else:
                rhs = F_target.subs(targets_to_arrays(arrays['x'], targets))
                rhs = rhs.subs(self.time_mapper) * volume
        return as_tuple(Eq(arrays['f'], rhs, subdomain=eq.subdomain))

    def _make_b(self, eq, b):
        rhs = 0. if isinstance(eq, EssentialBC) else b.subs(self.time_mapper)
        rhs = rhs * self.target.grid.symbolic_volume_cell
        return as_tuple(Eq(self.arrays['b'], rhs, subdomain=eq.subdomain))

    def _scale_bcs(self, eq, scdiag=None):
        """
        Scale ZeroRow equations using scdiag
        """
        scdiag = scdiag or self.scdiag
        return eq._rebuild(rhs=scdiag * eq.rhs) if isinstance(eq, ZeroRow) else eq


class MixedResidual(Residual):
    """
    """
    def __init__(self, target_eqns, arrays, time_mapper, scdiag):
        self.targets = as_tuple(target_eqns.keys())
        self.arrays = arrays
        self.time_mapper = time_mapper
        self.scdiag = scdiag
        self._build_equations(target_eqns)

    @property
    def formrhs(self):
        """
        """
        return None

    def _build_equations(self, target_eqns):
        all_formfuncs = []
        for target, eqns in target_eqns.items():

            formfuncs = chain.from_iterable(
                self._build_function_eq(eq, target)
                for eq in as_tuple(eqns)
            )
            all_formfuncs.extend(formfuncs)

        self._formfuncs = tuple(sorted(
            all_formfuncs, key=lambda e: not isinstance(e, EssentialBC)
        ))

    def _build_function_eq(self, eq, target):
        zeroed = eq.lhs - eq.rhs

        zeroed_eqn = Eq(eq.lhs - eq.rhs, 0)
        eval_zeroed_eqn = eval_time_derivatives(zeroed_eqn.lhs)

        volume = target.grid.symbolic_volume_cell

        mapper = {}
        for t in self.targets:
            target_funcs = set(generate_targets(Eq(eval_zeroed_eqn, 0), t))
            mapper.update(targets_to_arrays(self.arrays[t]['x'], target_funcs))

        if isinstance(eq, EssentialBC):
            rhs = (self.arrays[target]['x'] - eq.rhs)*self.scdiag[target]
            zero_row = ZeroRow(
                self.arrays[target]['f'], rhs, subdomain=eq.subdomain
            )
            zero_col = ZeroColumn(
                self.arrays[target]['x'], eq.rhs, subdomain=eq.subdomain
            )
            return (zero_row, zero_col)
        else:
            if isinstance(zeroed, (int, float)):
                rhs = zeroed * volume
            else:
                rhs = zeroed.subs(mapper)
                rhs = rhs.subs(self.time_mapper)*volume

        return as_tuple(Eq(self.arrays[target]['f'], rhs, subdomain=eq.subdomain))


class InitialGuess:
    """
    Enforce initial guess to satisfy essential BCs.
    # TODO: Extend this to "coupled".
    """
    def __init__(self, target, eqns, arrays):
        self.target = target
        self.eqns = as_tuple(eqns)
        self.arrays = arrays
        self._build_equations()

    @property
    def eqs(self):
        """
        """
        return self._eqs

    def _build_equations(self):
        """
        Return a list of initial guess equations.
        """
        self._eqs = [
            eq for eq in
            (self._make_initial_guess(e) for e in self.eqns)
            if eq is not None
        ]

    def _make_initial_guess(self, eq):
        if isinstance(eq, EssentialBC):
            assert eq.lhs == self.target
            return Eq(
                self.arrays['x'], eq.rhs,
                subdomain=eq.subdomain
            )
        else:
            return None


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

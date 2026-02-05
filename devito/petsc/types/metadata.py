import sympy
from itertools import chain
from functools import cached_property

from devito.tools import Reconstructable, sympy_mutex, as_tuple, frozendict
from devito.tools.dtypes_lowering import dtype_mapper
from devito.symbolics.extraction import separate_eqn, generate_targets, centre_stencil
from devito.types.equation import Eq
from devito.operations.solve import eval_time_derivatives

from devito.petsc.config import petsc_variables
from devito.petsc.types.object import Counter
from devito.petsc.types.equation import (
    EssentialBC, ZeroRow, ZeroColumn, NoOfEssentialBC, PointEssentialBC
)


class MetaData(sympy.Function, Reconstructable):
    def __new__(cls, expr, **kwargs):
        with sympy_mutex:
            obj = sympy.Function.__new__(cls, expr)
        obj.expr = expr
        return obj


class Initialize(MetaData):
    pass


class Finalize(MetaData):
    pass


class GetArgs(MetaData):
    pass


class SolverMetaData(MetaData):
    """
    A symbolic expression passed through the Operator, containing the metadata
    needed to execute the PETSc solver.
    """
    __rargs__ = ('expr',)
    __rkwargs__ = ('solver_parameters', 'field_data', 'time_mapper',
                   'localinfo', 'user_prefix', 'formatted_prefix',
                   'get_info')

    def __new__(cls, expr, solver_parameters=None,
                field_data=None, time_mapper=None, localinfo=None,
                user_prefix=None, formatted_prefix=None,
                get_info=None, **kwargs):

        with sympy_mutex:
            if isinstance(expr, tuple):
                expr = sympy.Tuple(*expr)
            obj = sympy.Basic.__new__(cls, expr)

        obj.expr = expr
        obj.solver_parameters = frozendict(solver_parameters)
        obj.field_data = field_data if field_data else FieldData()
        obj.time_mapper = time_mapper
        obj.localinfo = localinfo
        obj.user_prefix = user_prefix
        obj.formatted_prefix = formatted_prefix
        obj.get_info = get_info if get_info is not None else []
        return obj

    def __repr__(self):
        return f"{self.__class__.__name__}{self.expr}"

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    __hash__ = sympy.Basic.__hash__

    def _hashable_content(self):
        return (self.expr, self.formatted_prefix, self.solver_parameters)

    def __eq__(self, other):
        return (isinstance(other, SolverMetaData) and
                self.expr == other.expr and
                self.formatted_prefix == other.formatted_prefix
                and self.solver_parameters == other.solver_parameters)

    @property
    def grid(self):
        return self.field_data.grid

    @classmethod
    def eval(cls, *args):
        return None

    func = Reconstructable._rebuild


class LinearSolverMetaData(SolverMetaData):
    """
    Linear problems are handled by setting the SNESType to 'ksponly',
    enabling a unified interface for both linear and nonlinear solvers.
    """
    pass


class NonLinearSolverMetaData(SolverMetaData):
    """
    TODO: Non linear solvers are not yet supported.
    """
    pass


class FieldData:
    """
    Metadata for a single `target` field passed to `SolverMetaData`.
    Used to interface with PETSc SNES solvers at the IET level.

    Parameters
    ----------

    target : Function-like
        The target field to solve into, which is a Function-like object.
    jacobian : Jacobian
        Defines the matrix-vector product for the linear system, where the vector is
        the PETScArray representing the `target`.
    residual : Residual
        Defines the residual function F(target) = 0.
    initial_guess : InitialGuess
        Defines the initial guess for the solution, which satisfies
        essential boundary conditions.
    constrain_bc : ConstrainBC
        Defines the metadata for constraining essential boundary conditions i.e
        removing them from the global solve.
    arrays : dict
        A dictionary mapping `target` to its corresponding PETScArrays.
    """
    def __init__(self, target=None, jacobian=None, residual=None,
                 initial_guess=None, constrain_bc=None, arrays=None, **kwargs):
        self._target = target
        petsc_precision = dtype_mapper[petsc_variables['PETSC_PRECISION']]
        if self._target.dtype != petsc_precision:
            raise TypeError(
                "Your target dtype must match the precision of your "
                "PETSc configuration. "
                f"Expected {petsc_precision}, but got {self._target.dtype}."
            )
        self._jacobian = jacobian
        self._residual = residual
        self._initial_guess = initial_guess
        self._constrain_bc = constrain_bc
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
    def initial_guess(self):
        return self._initial_guess

    @property
    def constrain_bc(self):
        return self._constrain_bc

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
    """
    Metadata class passed to `SolverMetaData`, for mixed-field problems,
    where the solution vector spans multiple `targets`. Used to interface
    with PETSc SNES solvers at the IET level.

    Parameters
    ----------
    targets : list of Function-like
        The fields to solve into, each represented by a Function-like object.
    jacobian : MixedJacobian
        Defines the matrix-vector products for the full system Jacobian.
    residual : MixedResidual
        Defines the residual function F(targets) = 0.
    initial_guess : InitialGuess
        Defines the initial guess metadata, which satisfies
        essential boundary conditions.
    arrays : dict
        A dictionary mapping the `targets` to their corresponding PETScArrays.
    """
    def __init__(self, targets, arrays, jacobian=None, residual=None, constrain_bc=None):
        self._targets = as_tuple(targets)
        self._arrays = arrays
        self._jacobian = jacobian
        self._residual = residual
        self._constrain_bc = constrain_bc

    @cached_property
    def space_dimensions(self):
        space_dims = {t.space_dimensions for t in self.targets}
        if len(space_dims) > 1:
            # TODO: This may not actually have to be the case, but enforcing it for now
            raise ValueError(
                "All targets within a `petscsolve` call must have the"
                " same space dimensions."
            )
        return space_dims.pop()

    @cached_property
    def grid(self):
        """The unique `Grid` associated with all targets."""
        grids = [t.grid for t in self.targets]
        if len(set(grids)) > 1:
            raise ValueError(
                "Multiple `Grid`s detected in `petscsolve`;"
                " all targets must share one `Grid`."
            )
        return grids.pop()

    @cached_property
    def space_order(self):
        # NOTE: since we use DMDA to create vecs for the coupled solves,
        # all fields must have the same space order
        # ... re think this? limitation. For now, just force the
        # space order to be the same.
        # This isn't a problem for segregated solves.
        space_orders = [t.space_order for t in self.targets]
        if len(set(space_orders)) > 1:
            raise ValueError(
                "All targets within a `petscsolve` call must have the same space order."
            )
        return space_orders.pop()

    @property
    def targets(self):
        return self._targets


class BaseJacobian:
    def __init__(self, arrays, target=None):
        self.arrays = arrays
        self.target = target

    def _scale_non_bcs(self, matvecs, target=None):
        """
        Scale the symbolic expressions `matvecs` by the grid cell volume,
        excluding EssentialBCs.
        """
        target = target or self.target
        vol = target.grid.symbolic_volume_cell

        return [
            m if isinstance(m, EssentialBC) else m._rebuild(rhs=m.rhs * vol)
            for m in matvecs
        ]

    # TODO: rethink : can likely turn off if user requests to constrain bcs
    def _scale_bcs(self, matvecs, scdiag):
        """
        Scale the EssentialBCs in `matvecs` by `scdiag`.
        """
        return [
            m._rebuild(rhs=m.rhs * scdiag) if isinstance(m, ZeroRow) else m
            for m in matvecs
        ]

    def _compute_scdiag(self, matvecs, col_target=None):
        """
        Compute the diagonal scaling factor from the symbolic matrix-vector
        expressions in `matvecs`. If the centre stencil (i.e. the diagonal term of the
        matrix) is not unique, defaults to 1.0.
        """
        x = self.arrays[col_target or self.target]['x']

        centres = {
            centre_stencil(m.rhs, x, as_coeff=True)
            for m in matvecs if not isinstance(m, EssentialBC)
        }
        return centres.pop() if len(centres) == 1 else 1.0

    def _build_matvec_expr(self, expr, **kwargs):
        col_target = kwargs.get('col_target', self.target)
        row_target = kwargs.get('row_target', self.target)

        _, F_target, _, targets = separate_eqn(expr, col_target)
        if F_target:
            return self._make_matvec(
                expr, F_target, targets, col_target, row_target
            )
        else:
            return (None,)

    def _make_matvec(self, expr, F_target, targets, col_target, row_target):
        y = self.arrays[row_target]['y']
        x = self.arrays[col_target]['x']

        if isinstance(expr, EssentialBC):
            # NOTE: Essential BCs are trivial equations in the solver.
            # See `EssentialBC` for more details.
            zero_row = ZeroRow(y, x, subdomain=expr.subdomain)
            zero_column = ZeroColumn(x, 0., subdomain=expr.subdomain)
            return (zero_row, zero_column)
        else:
            rhs = F_target.subs(targets_to_arrays(x, targets))
            rhs = rhs.subs(self.time_mapper)
            return (Eq(y, rhs, subdomain=expr.subdomain),)


class Jacobian(BaseJacobian):
    """
    Represents a Jacobian matrix.

    This Jacobian is defined implicitly via matrix-vector products
    derived from the symbolic expressions provided in `matvecs`.

    The class assumes the problem is linear, meaning the Jacobian
    corresponds to a constant coefficient matrix and does not
    require explicit symbolic differentiation.
    """
    def __init__(self, target, exprs, arrays, time_mapper):
        super().__init__(arrays=arrays, target=target)
        self.exprs = exprs
        self.time_mapper = time_mapper
        self._build_matvecs()

    @property
    def matvecs(self):
        # TODO: add shortcut explanation etc
        """
        Stores the expressions used to generate the `MatMult` callback generated
        at the IET level. This function is passed to PETSc via
        `MatShellSetOperation(...,MATOP_MULT,(void (*)(void))MatMult)`.
        """
        return self._matvecs

    @property
    def scdiag(self):
        return self._scdiag

    @property
    def row_target(self):
        return self.target

    @property
    def col_target(self):
        return self.target

    @property
    def zero_memory(self):
        return True

    def _build_matvecs(self):
        matvecs = []
        for eq in self.exprs:
            matvecs.extend(
                e for e in self._build_matvec_expr(eq) if e is not None
            )
        key = lambda e: not isinstance(e, EssentialBC)
        matvecs = tuple(sorted(matvecs, key=key))

        matvecs = self._scale_non_bcs(matvecs)
        scdiag = self._compute_scdiag(matvecs)
        matvecs = self._scale_bcs(matvecs, scdiag)

        self._matvecs = matvecs
        self._scdiag = scdiag


class MixedJacobian(BaseJacobian):
    """
    Represents a Jacobian for a linear system with a solution vector
    composed of multiple fields (targets).

    Defines matrix-vector products for each sub-block,
    each with its own generated callback. The matrix may be treated
    as monolithic or block-structured in PETSc, but sub-block
    callbacks are generated in both cases.

    Assumes a linear problem, so this Jacobian corresponds to a
    coefficient matrix and does not require differentiation.

    # TODO: pcfieldsplit support for each block
    """
    def __init__(self, target_exprs, arrays, time_mapper):
        super().__init__(arrays=arrays, target=None)
        self.targets = tuple(target_exprs)
        self.time_mapper = time_mapper
        self._submatrices = []
        self._build_blocks(target_exprs)

    @property
    def submatrices(self):
        """
        Return a list of all submatrix blocks.
        Each block contains metadata about the matrix-vector products.
        """
        return self._submatrices

    @property
    def n_submatrices(self):
        """Return the number of submatrix blocks."""
        return len(self._submatrices)

    @cached_property
    def nonzero_submatrices(self):
        """Return SubMatrixBlock objects that have non-empty matvecs."""
        return [m for m in self.submatrices if m.matvecs]

    @cached_property
    def target_scaler_mapper(self):
        """
        Map each row target to the scdiag of its corresponding
        diagonal subblock.
        """
        mapper = {}
        for m in self.submatrices:
            if m.row_idx == m.col_idx:
                mapper[m.row_target] = m.scdiag
        return mapper

    def _build_blocks(self, target_exprs):
        """
        Build all SubMatrixBlock objects for the Jacobian.
        """
        for i, row_target in enumerate(self.targets):
            exprs = target_exprs[row_target]
            for j, col_target in enumerate(self.targets):

                matvecs = self._build_submatrix_matvecs(exprs, row_target, col_target)
                matvecs = [m for m in matvecs if m is not None]

                # Sort to put EssentialBC first if any
                matvecs = tuple(
                    sorted(matvecs, key=lambda e: not isinstance(e, EssentialBC))
                )
                matvecs = self._scale_non_bcs(matvecs, row_target)
                scdiag = self._compute_scdiag(matvecs, col_target)
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

    def _build_submatrix_matvecs(self, exprs, row_target, col_target):
        matvecs = []
        for expr in exprs:
            matvecs.extend(
                e for e in self._build_matvec_expr(
                    expr, col_target=col_target, row_target=row_target
                )
            )
        return matvecs

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
        return f"<MixedJacobian with {self.n_submatrices} submatrices: [{summary}]>"


class SubMatrixBlock:
    def __init__(self, name, matvecs, scdiag, row_target,
                 col_target, row_idx, col_idx, linear_idx):
        self.name = name
        self.matvecs = matvecs
        self.scdiag = scdiag
        self.row_target = row_target
        self.col_target = col_target
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.linear_idx = linear_idx

    @property
    def zero_memory(self):
        return False

    def __repr__(self):
        return (f"<SubMatrixBlock {self.name}>")


class Residual:
    """
    Generates the metadata needed to define the nonlinear residual function
    F(target) = 0 for use with PETSc's SNES interface.

    PETSc's SNES interface includes methods for solving nonlinear systems of
    equations using Newton-type methods. For linear problems, `SNESKSPONLY`
    can be used to perform a single Newton iteration, unifying the
    interface for both linear and nonlinear problems.

    This class encapsulates the symbolic expressions used to construct the
    residual function F(target) = F_(target) - b, where b contains all
    terms independent of the solution `target`.

    References:
        - https://petsc.org/main/manual/snes/
    """
    def __init__(self, target, exprs, arrays, time_mapper, scdiag):
        self.target = target
        self.exprs = exprs
        self.arrays = arrays
        self.time_mapper = time_mapper
        self.scdiag = scdiag
        self._build_exprs()

    @property
    def F_exprs(self):
        """
        Stores the expressions used to build the `FormFunction`
        callback generated at the IET level. This function is
        passed to PETSc via `SNESSetFunction(..., FormFunction, ...)`.
        """
        return self._F_exprs

    @property
    def b_exprs(self):
        """
        Stores the expressions used to generate the RHS
        vector `b` through the `FormRHS` callback generated at the IET level.
        The SNES solver is then called via `SNESSolve(..., b, target)`.
        """
        return self._b_exprs

    def _build_exprs(self):
        """
        """
        F_exprs = []
        b_exprs = []

        for e in self.exprs:
            b, F_target, _, targets = separate_eqn(e, self.target)
            F_exprs.extend(self._make_F_target(e, F_target, targets))
            # TODO: If b is zero then don't need a rhs vector+callback
            b_exprs.extend(self._make_b(e, b))

        self._F_exprs = tuple([self._scale_bcs(e) for e in F_exprs])
        self._b_exprs = tuple(b_exprs)

    def _make_F_target(self, eq, F_target, targets):
        arrays = self.arrays[self.target]
        volume = self.target.grid.symbolic_volume_cell

        # TODO: rethink : can likely turn off if user requests to constrain bcs
        if isinstance(eq, EssentialBC):
            # The initial guess satisfies the essential BCs, so this term is zero.
            # Still included to support Jacobian testing via finite differences.
            rhs = arrays['x'] - eq.rhs
            zero_row = ZeroRow(
                arrays['f'], rhs.subs(self.time_mapper), subdomain=eq.subdomain
            )
            # Move essential boundary condition to the right-hand side
            zero_col = ZeroColumn(
                arrays['x'], eq.rhs.subs(self.time_mapper), subdomain=eq.subdomain
            )
            return (zero_row, zero_col)

        else:
            if isinstance(F_target, (int, float)):
                rhs = F_target * volume
            else:
                rhs = F_target.subs(targets_to_arrays(arrays['x'], targets))
                rhs = rhs.subs(self.time_mapper) * volume
        return (Eq(arrays['f'], rhs, subdomain=eq.subdomain),)

    def _make_b(self, expr, b):
        b_arr = self.arrays[self.target]['b']
        # TODO: rethink : can likely turn off if user requests to constrain bcs
        rhs = 0. if isinstance(expr, EssentialBC) else b.subs(self.time_mapper)
        rhs = rhs * self.target.grid.symbolic_volume_cell
        return (Eq(b_arr, rhs, subdomain=expr.subdomain),)

    def _scale_bcs(self, eq, scdiag=None):
        """
        Scale ZeroRow exprs using scdiag
        """
        scdiag = scdiag or self.scdiag
        return eq._rebuild(rhs=scdiag * eq.rhs) if isinstance(eq, ZeroRow) else eq


class MixedResidual(Residual):
    """
    Generates the metadata needed to define the nonlinear residual function
    F(targets) = 0 for use with PETSc's SNES interface.
    """
    def __init__(self, target_exprs, arrays, time_mapper, scdiag):
        self.targets = tuple(target_exprs.keys())
        self.arrays = arrays
        self.time_mapper = time_mapper
        self.scdiag = scdiag
        self._build_exprs(target_exprs)

    @property
    def b_exprs(self):
        """
        For mixed solvers, a callback to form the RHS vector `b` is not generated,
        a single residual callback is generated to compute F(targets).
        TODO: Investigate if this is optimal.
        """
        return None

    def _build_exprs(self, target_exprs):
        residual_exprs = []
        for t, exprs in target_exprs.items():

            residual_exprs.extend(
                chain.from_iterable(
                    self._build_residual(e, t) for e in as_tuple(exprs)
                )
            )
        self._F_exprs = tuple(sorted(
            residual_exprs, key=lambda e: not isinstance(e, EssentialBC)
        ))

    def _build_residual(self, expr, target):
        zeroed = expr.lhs - expr.rhs
        zeroed_eqn = Eq(zeroed, 0)
        eval_zeroed_eqn = eval_time_derivatives(zeroed_eqn.lhs)

        volume = target.grid.symbolic_volume_cell

        mapper = {}
        for t in self.targets:
            target_funcs = set(generate_targets(Eq(eval_zeroed_eqn, 0), t))
            mapper.update(targets_to_arrays(self.arrays[t]['x'], target_funcs))

        # TODO: rethink : can likely turn off if user requests to constrain bcs
        if isinstance(expr, EssentialBC):
            rhs = (self.arrays[target]['x'] - expr.rhs)*self.scdiag[target]
            zero_row = ZeroRow(
                self.arrays[target]['f'], rhs, subdomain=expr.subdomain
            )
            zero_col = ZeroColumn(
                self.arrays[target]['x'], expr.rhs, subdomain=expr.subdomain
            )
            return (zero_row, zero_col)

        else:
            if isinstance(zeroed, (int, float)):
                rhs = zeroed * volume
            else:
                rhs = zeroed.subs(mapper)
                rhs = rhs.subs(self.time_mapper)*volume

        return (Eq(self.arrays[target]['f'], rhs, subdomain=expr.subdomain),)


class InitialGuess:
    """
    Metadata passed to `SolverExpr` to define the initial guess
    symbolic expressions, enforcing the initial guess to satisfy essential
    boundary conditions.
    """
    def __init__(self, target, exprs, arrays, time_mapper):
        self.target = target
        self.arrays = arrays
        self.time_mapper = time_mapper
        self._build_exprs(as_tuple(exprs))

    @property
    def exprs(self):
        return self._exprs

    def _build_exprs(self, exprs):
        """
        Return a list of initial guess symbolic expressions
        that satisfy essential boundary conditions.
        """
        self._exprs = tuple([
            eq for eq in
            (self._make_initial_guess(e) for e in exprs)
            if eq is not None
        ])

    def _make_initial_guess(self, expr):
        if isinstance(expr, EssentialBC):
            assert expr.lhs == self.target
            return Eq(
                self.arrays[self.target]['x'], expr.rhs.subs(self.time_mapper),
                subdomain=expr.subdomain
            )
        else:
            return None


class ConstrainBC:
    """
    Metadata passed to `SolverExpr` to constrain essential
    boundary conditions using a PetscSection.
    """
    def __init__(self, target, exprs, arrays):
        self.target = target
        self.arrays = arrays
        self._make_increment_exprs(as_tuple(exprs))
        self._make_point_bc_exprs(as_tuple(exprs))

    @property
    def increment_exprs(self):
        return self._increment_exprs

    @property
    def point_bc_exprs(self):
        return self._point_bc_exprs

    def _make_increment_exprs(self, exprs):
        """
        Return a list of symbolic expressions
        used to count the number of essential boundary conditions.
        """
        self._increment_exprs = tuple([
            eq for eq in
            (self._make_increment_expr(e) for e in exprs)
            if eq is not None
        ])

    def _make_increment_expr(self, expr):
        """
        Make the Eq that is used to increment the number of essential
        boundary nodes in the generated ccode.
        """
        if isinstance(expr, EssentialBC):
            assert expr.lhs == self.target
            return NoOfEssentialBC(
                Counter, 1,
                subdomain=expr.subdomain,
                implicit_dims=expr.subdomain.dimensions,
                target=self.target
            )
        else:
            return None

    def _make_point_bc_exprs(self, exprs):
        """
        Return a list of symbolic expressions
        used to count the number of essential boundary conditions.
        """
        self._point_bc_exprs = tuple([
            eq for eq in
            (self._make_point_bc_expr(e) for e in exprs)
            if eq is not None
        ])

    def _make_point_bc_expr(self, expr):
        """
        Make the Eq that is used to increment the number of essential
        boundary nodes in the generated ccode.
        """
        if isinstance(expr, EssentialBC):
            assert expr.lhs == self.target
            return PointEssentialBC(
                Counter, self.target,
                subdomain=expr.subdomain,
                target=self.target
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
    return frozendict(zip(targets, array_targets))

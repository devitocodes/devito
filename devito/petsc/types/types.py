import sympy

from itertools import chain

from devito.tools import Reconstructable, sympy_mutex, as_tuple
from devito.tools.dtypes_lowering import dtype_mapper
from devito.petsc.utils import petsc_variables
from devito.symbolics.extraction import separate_eqn, generate_targets
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
        return (self.target,)


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
    def jacobian(self):
        return self._jacobian

    @property
    def targets(self):
        return self._targets



class BaseJacobian:
    def __init__(self, targets, time_mapper, arrays):
        self.targets = as_tuple(targets)
        self.time_mapper = time_mapper
        self.arrays = arrays
        self.submatrices = self._initialize_submatrices()

    def _initialize_submatrices(self):
        """
        Create a dict of submatrices for each target with metadata.
        """
        submatrices = {}
        num_targets = len(self.targets)

        for i, target in enumerate(self.targets):
            submatrices[target] = {}
            for j in range(num_targets):
                key = f'J{i}{j}'
                submatrices[target][key] = {
                    'matvecs': None,
                    'derivative_wrt': self.targets[j],
                    'index': i * num_targets + j
                }

        return submatrices

    @property
    def submatrix_keys(self):
        """
        Return a list of all submatrix keys (e.g., ['J00', 'J01', 'J10', 'J11']).
        """
        return [key for submats in self.submatrices.values() for key in submats.keys()]

    @property
    def nonzero_submatrix_keys(self):
        """
        Returns a list of submats where 'matvecs' is not None.
        """
        return [
            key
            for submats in self.submatrices.values()
            for key, value in submats.items()
            if value['matvecs'] is not None
        ]

    @property
    def submat_to_index(self):
        """
        Returns a dict mapping submatrix keys to their index.
        """
        return {
            key: value['index']
            for submats in self.submatrices.values()
            for key, value in submats.items()
        }

    @property
    def diagonal_submatrix_keys(self):
        """
        Return a list of diagonal submatrix keys (e.g., ['J00', 'J11']).
        """
        keys = []
        for i, target in enumerate(self.targets):
            diag_key = f'J{i}{i}'
            if diag_key in self.submatrices[target]:
                keys.append(diag_key)
        return keys

    def set_submatrix(self, field, key, matvecs):
        """
        Set a specific submatrix for a field.

        Parameters
        ----------
        field : Function
            The target field that the submatrix operates on.
        key: str
            The identifier for the submatrix (e.g., 'J00', 'J01').
        matvecs: list of Eq
            The matrix-vector equations forming the submatrix.
        """
        if field in self.submatrices and key in self.submatrices[field]:
            self.submatrices[field][key]["matvecs"] = matvecs
        else:
            raise KeyError(f'Invalid field ({field}) or submatrix key ({key})')

    def get_submatrix(self, field, key):
        """
        Retrieve a specific submatrix.
        """
        return self.submatrices.get(field, {}).get(key, None)

    def build_matvec_eq(self, eq, target, arrays):
        b, F_target, _, targets = separate_eqn(eq, target)
        if F_target:
            return self.make_matvec(eq, F_target, targets, arrays)
        return (None,)

    def make_matvec(self, eq, F_target, targets, arrays):
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
            # rhs = rhs.subs(self.time_mapper) * self.cell_area
            rhs = rhs = rhs.subs(self.time_mapper)

        return as_tuple(Eq(arrays['y'], rhs, subdomain=eq.subdomain))

    def __repr__(self):
        return str(self.submatrices)


class Jacobian(BaseJacobian):

    @property
    def target(self):
        return self.targets[0]

    @property
    def matvecs(self):
        return self.submatrices[self.target]['J00']['matvecs']

    # TODO: use same structure arrays for both jacobian and mixedjacobian
    def build_block(self, eqns):
        for submat, mtvs in self.submatrices[self.target].items():
            matvecs = [
                e for eq in eqns for e in
                self.build_matvec_eq(eq, mtvs['derivative_wrt'], self.arrays)
            ]
            matvecs = [m for m in matvecs if m is not None]
            matvecs = tuple(sorted(matvecs, key=lambda e: not isinstance(e, EssentialBC)))

            if matvecs:
                self.set_submatrix(self.target, submat, matvecs)


class MixedJacobian(BaseJacobian):

    # TODO: use same structure arrays for both jacobian and mixedjacobian
    def build_block(self, target, eqns):
        for submat, mtvs in self.submatrices[target].items():
            matvecs = [
                e for eq in eqns for e in
                self.build_matvec_eq(eq, mtvs['derivative_wrt'], self.arrays[target])
            ]
            matvecs = [m for m in matvecs if m is not None]
            matvecs = tuple(sorted(matvecs, key=lambda e: not isinstance(e, EssentialBC)))

            if matvecs:
                self.set_submatrix(target, submat, matvecs)

    def build_blocks(self, target_eqns):
        for target, eqns in target_eqns.items():
            self.build_block(target, eqns)


class BaseResidual:
    def scale_essential_bcs(self, equations):
        """
        """
        return [
            eq._rebuild(rhs=self.scale * eq.rhs) if isinstance(eq, ZeroRow) else eq
            for eq in equations
        ]


class Residual(BaseResidual):
    """
    """

    def __init__(self, target, time_mapper, arrays, scale):
        self.target = target
        self.time_mapper = time_mapper
        self.arrays = arrays
        self.scale = scale
        self.formfuncs = []
        self.formrhs = []

    def build_equations(self, eqns):
        """
        """
        for eq in eqns:
            b, F_target, _, targets = separate_eqn(eq, self.target)
            F_target = self.make_F_target(eq, F_target, targets)
            b = self.make_b(eq, b)
            self.formfuncs.extend(F_target)
            self.formrhs.extend(b)

        self.formfuncs = self.scale_essential_bcs(self.formfuncs)

    def make_F_target(self, eq, F_target, targets):
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

    def make_b(self, eq, b):
        rhs = 0. if isinstance(eq, EssentialBC) else b.subs(self.time_mapper)
        rhs = rhs * self.target.grid.symbolic_volume_cell
        return as_tuple(Eq(self.arrays['b'], rhs, subdomain=eq.subdomain))


class MixedResidual(BaseResidual):
    """
    """
    # TODO: change default and pass in correct scale
    def __init__(self, targets, time_mapper, arrays, scale=1.0):
        self.targets = as_tuple(targets)
        self.time_mapper = time_mapper
        self.arrays = arrays
        self.scale = scale
        self.formfuncs = []

    def build_equations(self, eqn_dict):
        all_formfuncs = []
        for target, eqns in eqn_dict.items():

            formfuncs = chain.from_iterable(
                self.build_function_eq(eq, target)
                for eq in as_tuple(eqns)
            )

            # scale, = self._diag_scale[arrays[target]['x']]
            # fix this
            # scale = 1.0
            all_formfuncs.extend(self.scale_essential_bcs(formfuncs))

        self.formfuncs = tuple(sorted(
            all_formfuncs, key=lambda e: not isinstance(e, EssentialBC)
        ))


    def build_function_eq(self, eq, target):
        zeroed = eq.lhs - eq.rhs

        zeroed_eqn = Eq(eq.lhs - eq.rhs, 0)
        eval_zeroed_eqn = eval_time_derivatives(zeroed_eqn.lhs)

        mapper = {}
        for t in self.targets:
            target_funcs = set(generate_targets(Eq(eval_zeroed_eqn, 0), t))
            mapper.update(targets_to_arrays(self.arrays[t]['x'], target_funcs))

        if isinstance(eq, EssentialBC):
            rhs = self.arrays[target]['x'] - eq.rhs
            zero_row = ZeroRow(
                self.arrays[target]['f'], rhs, subdomain=eq.subdomain
            )
            zero_col = ZeroColumn(
                self.arrays[target]['x'], eq.rhs, subdomain=eq.subdomain
            )
            return (zero_row, zero_col)
        else:
            if isinstance(zeroed, (int, float)):
                # rhs = zeroed * self.cell_area
                rhs = zeroed
            else:
                rhs = zeroed.subs(mapper)
                # rhs = rhs.subs(self.time_mapper)*self.cell_area
                rhs = rhs.subs(self.time_mapper)

        return as_tuple(Eq(self.arrays[target]['f'], rhs, subdomain=eq.subdomain))



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

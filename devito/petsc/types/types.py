import sympy

from devito.tools import Reconstructable, sympy_mutex
from devito.tools.dtypes_lowering import dtype_mapper
from devito.petsc.utils import petsc_variables


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
    def __init__(self, target=None, matvecs=None, formfuncs=None, formrhs=None,
                 initialguess=None, arrays=None, **kwargs):
        self._target = kwargs.get('target', target)

        petsc_precision = dtype_mapper[petsc_variables['PETSC_PRECISION']]
        if self._target.dtype != petsc_precision:
            raise TypeError(
                f"Your target dtype must match the precision of your "
                f"PETSc configuration. "
                f"Expected {petsc_precision}, but got {self._target.dtype}."
            )
        self._matvecs = matvecs
        self._formfuncs = formfuncs
        self._formrhs = formrhs
        self._initialguess = initialguess
        self._arrays = arrays

    @property
    def target(self):
        return self._target

    @property
    def matvecs(self):
        return self._matvecs

    @property
    def formfuncs(self):
        return self._formfuncs

    @property
    def formrhs(self):
        return self._formrhs

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
    def __init__(self, submatrices=None):
        self.field_data_list = []
        self._submatrices = submatrices

    def add_field_data(self, field_data):
        self.field_data_list.append(field_data)

    def get_field_data(self, target):
        for field_data in self.field_data_list:
            if field_data.target == target:
                return field_data
        raise ValueError(f"FieldData with target {target} not found.")
    pass

    @property
    def target(self):
        return None

    @property
    def targets(self):
        return tuple(field_data.target for field_data in self.field_data_list)

    @property
    def space_dimensions(self):
        space_dims = {field_data.space_dimensions for field_data in self.field_data_list}
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
    def submatrices(self):
        return self._submatrices


class SubMatrices:
    def __init__(self, targets):
        self.targets = targets
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

    def __repr__(self):
        return str(self.submatrices)

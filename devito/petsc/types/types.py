import sympy

from devito.tools import Reconstructable, sympy_mutex


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
    __rkwargs__ = ('target', 'solver_parameters', 'matvecs',
                   'formfuncs', 'formrhs', 'arrays', 'time_mapper',
                   'localinfo')

    defaults = {
        'ksp_type': 'gmres',
        'pc_type': 'jacobi',
        'ksp_rtol': 1e-5,  # Relative tolerance
        'ksp_atol': 1e-50,  # Absolute tolerance
        'ksp_divtol': 1e5,  # Divergence tolerance
        'ksp_max_it': 1e4  # Maximum iterations
    }

    def __new__(cls, expr, target=None, solver_parameters=None,
                matvecs=None, formfuncs=None, formrhs=None,
                arrays=None, time_mapper=None, localinfo=None, **kwargs):

        if solver_parameters is None:
            solver_parameters = cls.defaults
        else:
            for key, val in cls.defaults.items():
                solver_parameters[key] = solver_parameters.get(key, val)

        with sympy_mutex:
            obj = sympy.Function.__new__(cls, expr)

        obj._expr = expr
        obj._target = target
        obj._solver_parameters = solver_parameters
        obj._matvecs = matvecs
        obj._formfuncs = formfuncs
        obj._formrhs = formrhs
        obj._arrays = arrays
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
                self.expr == other.expr and
                self.target == other.target)

    @property
    def expr(self):
        return self._expr

    @property
    def target(self):
        return self._target

    @property
    def solver_parameters(self):
        return self._solver_parameters

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
    def arrays(self):
        return self._arrays

    @property
    def time_mapper(self):
        return self._time_mapper

    @property
    def localinfo(self):
        return self._localinfo

    @classmethod
    def eval(cls, *args):
        return None

    func = Reconstructable._rebuild

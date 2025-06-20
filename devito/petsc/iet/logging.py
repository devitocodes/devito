from devito.petsc.iet.utils import petsc_call
from devito.petsc.types import PetscInt, PetscInfo
from devito.symbolics import Byref, FieldFromPointer
from devito.ir.iet import DummyExpr, BlankLine


class BaseLogger:
    """
    Base class for PETSc loggers that collect solver related statistics.

    This class can be subclassed to emit different information based
    on the 'log-level' set in the Devito configuration.
    """
    function_mapper = {}

    def __init__(self, **kwargs):
        self.sobjs = kwargs.get('solver_objs')
        self.sreg = kwargs.get('sregistry')
        self.section_mapper = kwargs.get('section_mapper', {})
        self.injectsolve = kwargs.get('injectsolve', None)

        name = self.sreg.make_name(prefix='info')
        pname = self.sreg.make_name(prefix='petscstats')
        self.logobjs = self._generate_objs()

        self.statstruct = PetscInfo(
            name, pname, self.logobjs, self.sobjs,
            self.section_mapper, self.injectsolve,
            self.function_mapper,

        )
        self.calls = self._generate_calls()

    def _generate_objs(self):
        """
        Create PETSc objects to store metadata.
        """
        return {}

    def _generate_calls(self):
        """
        Generate the PETSc calls that will be injected into the C code to
        extract solver statistics.

        This method should be overridden by subclasses to implement log-level
        specific behaviour.
        """
        return ()


class PerfLogger(BaseLogger):
    """Logger for log-level 'PERF'"""

    function_mapper = {
        'KSPGetIterationNumber': ('kspits', PetscInt),
        'SNESGetIterationNumber': ('snesits', PetscInt)
    }

    def _generate_objs(self):
        """Create PETSc objects for the given function_mapper"""
        return {
            name: obj_type(self.sreg.make_name(prefix=name))
            for _, (name, obj_type) in self.function_mapper.items()
        }

    def _generate_calls(self):
        struct = self.statstruct
        calls = []
        for name in ['snes', 'ksp']:
            obj = self.sobjs[name]
            logobj = self.logobjs[f"{name}its"]

            calls.append(
                petsc_call(f"{name.upper()}GetIterationNumber", [obj, Byref(logobj)])
            )
            # TODO: Perform a PetscCIntCast here?
            expr = DummyExpr(FieldFromPointer(logobj._C_symbol, struct), logobj._C_symbol)
            calls.append(expr)

        calls.append(BlankLine,)
        return tuple(calls)


class DebugLogger(PerfLogger):
    """Logger for log-level 'DEBUG'.

    This is the most verbose logging level. It may be extended in the future
    to emit additional information beyond what is provided by the 'PERF' log level.
    """
    pass


petsc_logger_registry = {
    'DEBUG': DebugLogger,
    'PERF': PerfLogger
}

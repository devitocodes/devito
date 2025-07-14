from functools import cached_property

from devito.symbolics import Byref, FieldFromPointer
from devito.ir.iet import DummyExpr
from devito.logger import PERF

from devito.petsc.iet.utils import petsc_call
from devito.petsc.logging import petsc_return_variable_dict, PetscInfo


class PetscLogger:
    """
    Class for PETSc loggers that collect solver related statistics.
    """
    def __init__(self, level, **kwargs):
        self.sobjs = kwargs.get('solver_objs')
        self.sreg = kwargs.get('sregistry')
        self.section_mapper = kwargs.get('section_mapper', {})
        self.injectsolve = kwargs.get('injectsolve', None)

        self.function_list = []

        if level <= PERF:
            self.function_list.extend([
                'kspgetiterationnumber',
                'snesgetiterationnumber'
            ])

        # TODO: To be extended with if level <= DEBUG: ...

        name = self.sreg.make_name(prefix='petscinfo')
        pname = self.sreg.make_name(prefix='petscprofiler')

        self.statstruct = PetscInfo(
            name, pname, self.logobjs, self.sobjs,
            self.section_mapper, self.injectsolve,
            self.function_list
        )

    @cached_property
    def logobjs(self):
        """
        Create PETSc objects specifically needed for logging solver statistics.
        """
        return {
            info.name: info.variable_type(
                self.sreg.make_name(prefix=info.output_param)
            )
            for func_name in self.function_list
            for info in [petsc_return_variable_dict[func_name]]
        }

    @cached_property
    def calls(self):
        """
        Generate the PETSc calls that will be injected into the C code to
        extract solver statistics.
        """
        struct = self.statstruct
        calls = []
        for param in self.function_list:
            param = petsc_return_variable_dict[param]

            inputs = []
            for i in param.input_params:
                inputs.append(self.sobjs[i])

            logobj = self.logobjs[param.name]

            calls.append(
                petsc_call(param.name, inputs + [Byref(logobj)])
            )
            # TODO: Perform a PetscCIntCast here?
            expr = DummyExpr(FieldFromPointer(logobj._C_symbol, struct), logobj._C_symbol)
            calls.append(expr)

        return tuple(calls)

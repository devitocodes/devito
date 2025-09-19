from functools import cached_property

from devito.symbolics import Byref, FieldFromPointer
from devito.ir.iet import DummyExpr
from devito.logger import PERF
from devito.tools import frozendict

from devito.petsc.iet.utils import petsc_call
from devito.petsc.logging import petsc_return_variable_dict, PetscInfo


class PetscLogger:
    """
    Class for PETSc loggers that collect solver related statistics.
    """
    # TODO: Update docstring with kwargs
    def __init__(self, level, **kwargs):

        self.query_functions = kwargs.get('get_info', [])
        self.sobjs = kwargs.get('solver_objs')
        self.sreg = kwargs.get('sregistry')
        self.section_mapper = kwargs.get('section_mapper', {})
        self.inject_solve = kwargs.get('inject_solve', None)

        if level <= PERF:
            funcs = [
                # KSP specific
                'kspgetiterationnumber',
                'kspgettolerances',
                'kspgetconvergedreason',
                'kspgettype',
                'kspgetnormtype',
                # SNES specific
                'snesgetiterationnumber',
            ]
            self.query_functions = set(self.query_functions)
            self.query_functions.update(funcs)
            self.query_functions = sorted(list(self.query_functions))

        # TODO: To be extended with if level <= DEBUG: ...

        name = self.sreg.make_name(prefix='petscinfo')
        pname = self.sreg.make_name(prefix='petscprofiler')

        self.statstruct = PetscInfo(
            name, pname, self.petsc_option_mapper, self.sobjs,
            self.section_mapper, self.inject_solve,
            self.query_functions
        )

    @cached_property
    def petsc_option_mapper(self):
        """
        For each function in `self.query_functions`, look up its metadata in
        `petsc_return_variable_dict` and instantiate the corresponding PETSc logging
        variables with names from the symbol registry.

        Example:
        --------
        >>> self.query_functions
        ['kspgetiterationnumber', 'snesgetiterationnumber', 'kspgettolerances']

        >>> self.petsc_option_mapper
        {
            'KSPGetIterationNumber': {'kspits': kspits0},
            'KSPGetTolerances': {'rtol': rtol0, 'atol': atol0, ...}
        }
        """
        opts = {}
        for func_name in self.query_functions:
            info = petsc_return_variable_dict[func_name]
            opts[info.name] = {}
            for vtype, out in zip(info.variable_type, info.output_param, strict=True):
                opts[info.name][out] = vtype(self.sreg.make_name(prefix=out))
        return frozendict(opts)

    @cached_property
    def calls(self):
        """
        Generate the PETSc calls that will be injected into the C code to
        extract solver statistics.
        """
        struct = self.statstruct
        calls = []
        for func_name in self.query_functions:
            return_variable = petsc_return_variable_dict[func_name]

            input = self.sobjs[return_variable.input_params]
            output_params = self.petsc_option_mapper[return_variable.name].values()
            by_ref_output = [Byref(i) for i in output_params]

            calls.append(
                petsc_call(return_variable.name, [input] + by_ref_output)
            )
            # TODO: Perform a PetscCIntCast here?
            exprs = [
                DummyExpr(FieldFromPointer(i._C_symbol, struct), i._C_symbol)
                for i in output_params
            ]
            calls.extend(exprs)

        return tuple(calls)

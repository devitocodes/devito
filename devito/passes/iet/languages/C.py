import numpy as np
from sympy.printing.c import C99CodePrinter

from devito.ir import Call, BasePrinter
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB
from devito.symbolics import c_complex, c_double_complex
from devito.tools import dtype_to_cstr

from devito.petsc.utils import petsc_type_mappings
from devito.petsc.iet.passes import rebuild_child_user_struct, rebuild_parent_user_struct


__all__ = ['CBB', 'CDataManager', 'COrchestrator']


class CBB(LangBB):

    mapper = {
        # Misc
        'header-array': None,
        # Complex
        'includes-complex': 'complex.h',
        # Allocs
        'header-memcpy': 'string.h',
        'host-alloc': lambda i, j, k:
            Call('posix_memalign', (i, j, k)),
        'host-alloc-pin': lambda i, j, k:
            Call('posix_memalign', (i, j, k)),
        'host-free': lambda i:
            Call('free', (i,)),
        'host-free-pin': lambda i:
            Call('free', (i,)),
        'alloc-global-symbol': lambda i, j, k:
            Call('memcpy', (i, j, k)),
    }


class CDataManager(DataManager):
    langbb = CBB


class COrchestrator(Orchestrator):
    langbb = CBB


class CPrinter(BasePrinter, C99CodePrinter):

    _default_settings = {**BasePrinter._default_settings,
                         **C99CodePrinter._default_settings}
    _func_literals = {np.float32: 'f', np.complex64: 'f'}
    _func_prefix = {np.float32: 'f', np.float64: 'f',
                    np.complex64: 'c', np.complex128: 'c'}
    _includes = ['stdlib.h', 'math.h', 'sys/time.h']

    # These cannot go through _print_xxx because they are classes not
    # instances
    type_mappings = {**C99CodePrinter.type_mappings,
                     c_complex: 'float _Complex',
                     c_double_complex: 'double _Complex'}

    def _print_ImaginaryUnit(self, expr):
        return '_Complex_I'

    def _print_ListInitializer(self, expr):
        li = super()._print_ListInitializer(expr)
        if expr.dtype:
            # C99, unlike CXX, supports compound literals
            tstr = dtype_to_cstr(expr.dtype)
            return f'({tstr}[]){li}'
        else:
            return li

    def _print_ComplexPart(self, expr):
        return (f'{self.func_prefix(expr)}{expr._name}{self.func_literal(expr)}'
                f'({self._print(expr.args[0])})')

    def _print_Conj(self, expr):
        # In C, conj is not preceeded by the func_prefix
        return (f'conj{self.func_literal(expr)}({self._print(expr.args[0])})')


class PetscCPrinter(CPrinter):
    _restrict_keyword = ''

    type_mappings = {**CPrinter.type_mappings, **petsc_type_mappings}

    def _print_Pi(self, expr):
        return 'PETSC_PI'


class PetscCDataManager(CDataManager):
    def process(self, graph):
        """
        Apply the `place_definitions` and `place_casts` passes.

        These passes may introduce new symbols, which must be incorporated into
        the relevant PETSc structures. These structures are subsequently used by PETSc
        callback functions to access necessary information (via DMGetApplicationContext).

        After applying the passes, the method:
        1. Rebuilds the PETSc structures to include any new symbols.
        2. Updates the `PopulateUserContext` callback to populate the new fields.
        """
        self.place_definitions(graph, globs=set())
        self.place_casts(graph)

        callback_struct_mapper = {}
        # Rebuild the child user struct (`CallbackUserStruct`) - these structs are used
        # to access information in PETSc callback functions through
        # `DMGetApplicationContext`
        rebuild_child_user_struct(graph, mapper=callback_struct_mapper)
        # Update the parent user struct - these structs are registered in the main
        # kernel via `DMSetApplicationContext` and populated in the `PopulateUserContext`
        # callback
        rebuild_parent_user_struct(graph, mapper=callback_struct_mapper)

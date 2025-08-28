import numpy as np
from sympy.printing.c import C99CodePrinter

from devito.ir import Call, BasePrinter, FindSymbols
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB
from devito.passes.iet.engine import iet_pass
from devito.symbolics import c_complex, c_double_complex
from devito.tools import dtype_to_cstr
from devito.petsc.utils import petsc_type_mappings

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
    

# TODO: move all petsc stuff to petsc module
class PetscCDataManager(CDataManager):
    def process(self, graph):
        """
        Apply the `place_definitions` and `place_casts` passes.
        Also, ....
        """
        self.place_definitions(graph, globs=set())
        self.place_casts(graph)

        # Apply mapper if necessary to any symbols etc..
        # some new symbols may now appear in the efuncs (e.g., from casts)
        # so we need to update the callback that populates the struct and
        # then map them using a FieldFromPointer
        # from IPython import embed; embed()

        # from collections import defaultdict

        # symbol_map = defaultdict(set)

        # # step 1: find all user contexts
        # for name, efunc in graph.efuncs.items():
        #     if name.startswith("PopulateUserContext"):
        #         # step 2: extract the struct symbol (assume always 1 element)
        #         ctx_symbol = FindSymbols().visit(efunc)[0]

        #         # step 3: loop over other PETScCallables
        #         for other_name, other_efunc in graph.efuncs.items():
        #             if other_name == name:
        #                 continue
        #             if "PETScCallable" in str(type(other_efunc)):  # adjust if you have a better type check
        #                 symbols = FindSymbols().visit(other_efunc)
        #                 if ctx_symbol in symbols:
        #                     # step 4: collect all symbols into the set
        #                     from devito.types import Temp, TempArray
        #                     from devito.petsc.types.array import PETScArray
        #                     from devito.petsc.types.object import PetscObject
        #                     fields = [f.function for f in FindSymbols('basics').visit(other_efunc)]
        #                     avoid = (PETScArray, Temp, TempArray, PetscObject)
        #                     fields = [f for f in fields if not isinstance(f.function, avoid)]
        #                     fields = [
        #                         f for f in fields if not (f.is_Dimension and not (f.is_Time or f.is_Modulo))
        #                     ]

        #                     symbol_map[ctx_symbol].update(fields)

        # old_ctx_to_new_ctx_mapper = {}
        # for ctx_symbol, fields in symbol_map.items():
        #     new_struct = ctx_symbol._rebuild(fields=tuple(fields))
        #     old_ctx_to_new_ctx_mapper[ctx_symbol] = new_struct

        #     old_ctx_to_new_ctx_mapper[ctx_symbol] = new_struct
        
        # self.old_ctx_to_new_ctx_mapper = old_ctx_to_new_ctx_mapper

        # # update all efuncs with the new struct
        # self.replace_user_context(graph)

        # self.update_user_context_callback(graph)



    @iet_pass
    def replace_user_context(self, iet, **kwargs):

        # if not iet.name.startswith("PopulateUserContext"):
        #     return iet, {}

        # from IPython import embed; embed()
        # new_body = Uxreplace()
        from devito.ir.iet import Uxreplace, CallableBody
        new_body = Uxreplace(self.old_ctx_to_new_ctx_mapper).visit(iet.body)
        new_parameters = tuple(self.old_ctx_to_new_ctx_mapper.get(p, p) for p in iet.parameters)
        iet = iet._rebuild(body=new_body, parameters=new_parameters)
        # from IPython import embed; embed()

        # all_symbs = FindSymbols().visit(iet)
        # from devito.petsc.iet.passes import objs


        return iet, {}


    @iet_pass
    def update_user_context_callback(self, iet, **kwargs):

        if not iet.name.startswith("PopulateUserContext"):
            return iet, {}
        
        # Grab the context struct
        ctx = iet.parameters[0]
        from devito.symbolics import FieldFromPointer
        from devito.ir.iet import DummyExpr, Callable, List, CallableBody
        from devito.petsc.types.macros import petsc_func_begin_user

        body = [
            DummyExpr(FieldFromPointer(i._C_symbol, ctx), i._C_symbol)
            for i in ctx.callback_fields
        ]

        body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )
        iet = iet._rebuild(body=body)
        # from IPython import embed; embed()
        return iet, {}


        

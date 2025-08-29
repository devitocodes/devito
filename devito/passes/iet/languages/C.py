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


        # first of all, rebuild any `CallbackUserStruct` to include any new fields
        # that may have appeared from `place_definitions` and `place_casts`.
        callback_struct_mapper = {}

        self.rebuild_callback_struct(graph, mapper=callback_struct_mapper)

        main_struct_mapper = {}
        for ctx in callback_struct_mapper.values():
            main_struct = ctx.parent
            main_struct_mapper[main_struct] = main_struct._rebuild(fields=
                ctx.fields
            )

        # self.replace_user_context(graph, mapper=main_struct_mapper)

        # from IPython import embed; embed()


    @iet_pass
    def rebuild_callback_struct(self, iet, mapper, **kwargs):

        from devito.petsc.types.object import CallbackUserStruct
        # check to see if there are any `CallbackUserStruct` in iet

        callback_user_struct = set([i for i in FindSymbols().visit(iet) if isinstance(i, CallbackUserStruct)])

        # do nothing if there are no `CallbackUserStruct` in the efunc
        if not callback_user_struct:
            return iet, {}
        # There will only be one `CallbackUserStruct` per efunc, so pop it out
        # since 
        assert len(callback_user_struct) == 1
        callback_user_struct = callback_user_struct.pop()
        from devito.petsc.iet.utils import get_user_struct_fields

        fields = get_user_struct_fields(iet)
        new_fields = []
        for f in fields:
            if f not in callback_user_struct.fields:
                new_fields.append(f)

        all_fields = callback_user_struct.fields + new_fields
        mapper[callback_user_struct] = callback_user_struct._rebuild(fields=all_fields)
        from devito.ir.iet import Uxreplace, Dereference
        from devito.symbolics import FieldFromPointer

        new_body = Uxreplace(mapper).visit(iet.body)

        # De-reference any Scalar fields e.g. x_size
        from devito.types import Scalar
        derefs = tuple(
            [Dereference(i, mapper[callback_user_struct]) for i in
             new_fields if isinstance(i.function, Scalar)]
        )
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, callback_user_struct) for i in new_fields if not isinstance(i, Scalar)}
        new_body = Uxreplace(subs).visit(new_body)
        new_body = new_body._rebuild(standalones=new_body.standalones + derefs)
        
        iet = iet._rebuild(body=new_body)

        return iet, {}


    @iet_pass
    def replace_user_context(self, iet, mapper, **kwargs):

        from devito.ir.iet import Uxreplace
        if not iet.name.startswith("PopulateUserContext"):
            return iet._rebuild(body=Uxreplace(mapper).visit(iet.body)), {}
        
        # If it is the `PopulateUserContext` callback, then we need to
        # also update the body to populate the new fields in the struct
        # and also update the parameters
        from devito.petsc.types.object import MainUserStruct
        from devito.symbolics import FieldFromPointer
        from devito.ir.iet import DummyExpr
        old_user_ctx = [i for i in iet.parameters if isinstance(i, MainUserStruct)].pop()
        new_user_ctx = mapper[old_user_ctx]

        new_body = [
            DummyExpr(FieldFromPointer(i._C_symbol, new_user_ctx), i._C_symbol)
            for i in new_user_ctx.callback_fields
        ]

        new_body = iet.body._rebuild(body=new_body)

        iet = iet._rebuild(body=new_body, parameters=(new_user_ctx,))

        return iet, {}
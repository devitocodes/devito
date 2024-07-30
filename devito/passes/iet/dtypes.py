import ctypes
import numpy as np

from devito.arch.compiler import Compiler
from devito.ir import Callable, Dereference, FindSymbols, Node, SymbolRegistry, Uxreplace
from devito.passes.iet.langbase import LangBB
from devito.symbolics.extended_dtypes import Float16P
from devito.tools import as_list
from devito.types.basic import AbstractSymbol, Basic, Symbol

__all__ = ['lower_dtypes']


def lower_dtypes(iet: Callable, lang: type[LangBB], compiler: Compiler,
                 sregistry: SymbolRegistry) -> tuple[Callable, dict]:
    """
    Lowers float16 scalar types to pointers since we can't directly pass their
    value. Also includes headers for complex arithmetic if needed.
    """

    iet, metadata = _complex_includes(iet, lang, compiler)

    # Lower float16 parameters to pointers and dereference
    prefix: list[Node] = []
    params_mapper: dict[AbstractSymbol, AbstractSymbol] = {}
    body_mapper: dict[AbstractSymbol, Symbol] = {}

    params_set = set(iet.parameters)
    s: AbstractSymbol
    for s in FindSymbols('abstractsymbols').visit(iet):
        if s.dtype != np.float16 or s not in params_set:
            continue

        # Replace the parameter with a pointer; replace occurences in the IET
        # body with dereferenced symbol (using the original symbol's dtype)
        ptr: AbstractSymbol = s._rebuild(dtype=Float16P, is_const=True)
        val = Symbol(name=sregistry.make_name(prefix='hf'), dtype=s.dtype,
                     is_const=s.is_const)

        params_mapper[s], body_mapper[s] = ptr, val
        prefix.append(Dereference(val, ptr))  # val = *ptr

    # Apply the replacements
    prefix.extend(as_list(Uxreplace(body_mapper).visit(iet.body)))
    params: tuple[Basic] = Uxreplace(params_mapper).visit(iet.parameters)

    iet = iet._rebuild(body=prefix, parameters=params)
    return iet, metadata


def _complex_includes(iet: Callable, lang: type[LangBB],
                      compiler: Compiler) -> tuple[Callable, dict]:
    """
    Includes complex arithmetic headers for the given language, if needed.
    """

    # Check if there are complex numbers that always take dtype precedence
    types = {f.dtype for f in FindSymbols().visit(iet)
             if not issubclass(f.dtype, ctypes._Pointer)}

    if not any(np.issubdtype(d, np.complexfloating) for d in types):
        return iet, {}

    metadata = {}
    lib = (lang['header-complex'],)

    if lang.get('complex-namespace') is not None:
        metadata['namespaces'] = lang['complex-namespace']

    # Some languges such as c++11 need some extra arithmetic definitions
    if lang.get('def-complex'):
        dest = compiler.get_jit_dir()
        hfile = dest.joinpath('complex_arith.h')
        with open(str(hfile), 'w') as ff:
            ff.write(str(lang['def-complex']))
        lib += (str(hfile),)

    metadata['includes'] = lib

    return iet, metadata

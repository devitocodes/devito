import ctypes
import numpy as np

from devito.arch.compiler import Compiler
from devito.ir import Callable, FindSymbols, SymbolRegistry
from devito.passes.iet.langbase import LangBB

__all__ = ['lower_dtypes']


def lower_dtypes(iet: Callable, lang: type[LangBB], compiler: Compiler,
                 sregistry: SymbolRegistry) -> tuple[Callable, dict]:
    """
    Lowers float16 scalar types to pointers since we can't directly pass their
    value. Also includes headers for complex arithmetic if needed.
    """
    # Complex numbers
    iet, metadata = _complex_includes(iet, lang, compiler)

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

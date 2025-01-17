import ctypes
import numpy as np

from devito.arch.compiler import Compiler
from devito.ir import Callable, FindSymbols, SymbolRegistry
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.tools import as_tuple

__all__ = ['lower_dtypes']


@iet_pass
def _complex_includes(iet: Callable, lang: type[LangBB] = None,
                      compiler: Compiler = None) -> tuple[Callable, dict]:
    """
    Includes complex arithmetic headers for the given language, if needed.
    """

    # Check if there are complex numbers that always take dtype precedence
    types = set()
    for f in FindSymbols().visit(iet):
        try:
            if not issubclass(f.dtype, ctypes._Pointer):
                types.add(f.dtype)
        except TypeError:
            pass

    if not any(np.issubdtype(d, np.complexfloating) for d in types):
        return iet, {}

    metadata = {}
    lib = as_tuple(lang['header-complex'])

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


dtype_passes = [_complex_includes]


def lower_dtypes(graph: Callable, lang: type[LangBB] = None, compiler: Compiler = None,
                 sregistry: SymbolRegistry = None, **kwargs) -> tuple[Callable, dict]:
    """
    Lowers float16 scalar types to pointers since we can't directly pass their
    value. Also includes headers for complex arithmetic if needed.
    """

    for dtype_pass in dtype_passes:
        dtype_pass(graph, lang=lang, compiler=compiler)

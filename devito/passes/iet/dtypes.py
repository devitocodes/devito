import numpy as np

from devito.arch.compiler import Compiler
from devito.ir import Callable, SymbolRegistry
from devito.ir.iet.utils import has_dtype
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.tools import as_tuple

__all__ = ['lower_dtypes']


@iet_pass
def _complex_includes(iet: Callable, langbb: type[LangBB], compiler: Compiler,
                      sregistry: SymbolRegistry) -> tuple[Callable, dict]:
    """
    Includes complex arithmetic headers for the given language, if needed.
    """
    # Check if there are complex numbers that always take dtype precedence
    if not has_dtype(iet, np.complexfloating):
        return iet, {}

    metadata = {}
    lib = as_tuple(langbb['includes-complex'])

    if langbb.get('complex-namespace') is not None:
        metadata['namespaces'] = langbb['complex-namespace']

    # Some languges such as c++11 need some extra arithmetic definitions
    if langbb.get('def-complex'):
        dest = compiler.get_jit_dir()
        hfile = dest.joinpath('complex_arith.h')
        with open(str(hfile), 'w') as ff:
            ff.write(str(langbb['def-complex']))
        lib += (str(hfile),)

    metadata['includes'] = lib

    return iet, metadata


dtype_passes = [_complex_includes]


def lower_dtypes(graph: Callable,
                 langbb: type[LangBB] = None,
                 compiler: Compiler = None,
                 sregistry: SymbolRegistry = None, **kwargs) -> tuple[Callable, dict]:
    """
    Lowers float16 scalar types to pointers since we can't directly pass their
    value. Also includes headers for complex arithmetic if needed.
    """

    for dtype_pass in dtype_passes:
        dtype_pass(graph, langbb=langbb, compiler=compiler, sregistry=sregistry)

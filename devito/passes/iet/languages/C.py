import ctypes as ct
import numpy as np

from devito.ir import Call
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB
from devito.tools.dtypes_lowering import ctypes_vector_mapper


__all__ = ['CBB', 'CDataManager', 'COrchestrator']


class CCFloat(np.complex64):
    pass


class CCDouble(np.complex128):
    pass


c_complex = type('_Complex float', (ct.c_double,), {})
c_double_complex = type('_Complex double', (ct.c_longdouble,), {})

ctypes_vector_mapper[CCFloat] = c_complex
ctypes_vector_mapper[CCDouble] = c_double_complex


class CBB(LangBB):

    mapper = {
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
        # Complex
        'header-complex': 'complex.h',
        'types': {np.complex128: CCDouble, np.complex64: CCFloat},
    }


class CDataManager(DataManager):
    lang = CBB


class COrchestrator(Orchestrator):
    lang = CBB

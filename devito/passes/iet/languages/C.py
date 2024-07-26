import numpy as np

from devito.ir import Call
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB
from devito.symbolics.extended_dtypes import (Float16P, c_complex, c_double_complex,
                                              c_half, c_half_p)


__all__ = ['CBB', 'CDataManager', 'COrchestrator', 'c_float16', 'c_float16_p']


c99_complex = type('_Complex float', (c_complex,), {})
c99_double_complex = type('_Complex double', (c_double_complex,), {})

c_float16 = type('_Float16', (c_half,), {})
c_float16_p = type('_Float16 *', (c_half_p,), {'_type_': c_float16})


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
        # Complex and float16
        'header-complex': 'complex.h',
        'types': {np.complex128: c99_double_complex,
                  np.complex64: c99_complex,
                  np.float16: c_float16,
                  Float16P: c_float16_p}
    }


class CDataManager(DataManager):
    lang = CBB


class COrchestrator(Orchestrator):
    lang = CBB

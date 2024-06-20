import numpy as np

from devito.ir import Call
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB
from devito.tools import CustomNpType

__all__ = ['CBB', 'CDataManager', 'COrchestrator']


CCFloat = CustomNpType('_Complex float', np.complex64)
CCDouble = CustomNpType('_Complex double', np.complex128)


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

import numpy as np

from devito.ir import Call
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB
from devito.symbolics.extended_dtypes import c_complex, c_double_complex
from devito.symbolics.printer import _DevitoPrinterBase

__all__ = ['CBB', 'CDataManager', 'COrchestrator']


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
    }


class CDataManager(DataManager):
    lang = CBB


class COrchestrator(Orchestrator):
    lang = CBB


class CDevitoPrinter(_DevitoPrinterBase):

    # These cannot go through _print_xxx because they are classes not
    # instances
    type_mappings = {**_DevitoPrinterBase.type_mappings,
                     c_complex: 'float _Complex',
                     c_double_complex: 'double _Complex'}

    _func_prefix = {**_DevitoPrinterBase._func_prefix, np.complex64: 'c',
                    np.complex128: 'c'}

    def _print_ImaginaryUnit(self, expr):
        return '_Complex_I'

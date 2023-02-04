from devito.ir import Call
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB

__all__ = ['CBB', 'CDataManager', 'COrchestrator']


class CBB(LangBB):

    mapper = {
        'header-memcpy': 'string.h',
        'host-alloc': lambda i, j, k:
            Call('posix_memalign', (i, j, k)),
        'host-free': lambda i:
            Call('free', (i,)),
        'alloc-global-symbol': lambda i, j, k:
            Call('memcpy', (i, j, k))
    }


class CDataManager(DataManager):
    lang = CBB


class COrchestrator(Orchestrator):
    lang = CBB

from devito.ir import Call
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.langbase import LangBB

__all__ = ['CBB', 'CDataManager']


class CBB(LangBB):

    mapper = {
        'aligned': lambda i:
            '__attribute__((aligned(%d)))' % i,
        'host-alloc': lambda i, j, k:
            'posix_memalign((void**)&%s, %d, %s)' % (i, j, k),
        'host-free': lambda i:
            Call('free', (i,)),
    }


class CDataManager(DataManager):
    lang = CBB

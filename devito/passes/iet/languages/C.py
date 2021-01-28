from devito.passes.iet.definitions import DataManager
from devito.passes.iet.langbase import LangBB

__all__ = ['CBB', 'CDataManager']


class CBB(LangBB):

    mapper = {
        'aligned': lambda i:
            '__attribute__((aligned(%d)))' % i,
        'alloc-host': lambda i, j, k:
            'posix_memalign((void**)&%s, %d, %s)' % (i, j, k),
        'free-host': lambda i:
            'free(%s)' % i,
    }


class CDataManager(DataManager):
    lang = CBB

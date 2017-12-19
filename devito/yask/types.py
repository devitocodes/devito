from devito.logger import yask as log
import devito.types as types

from devito.yask.wrappers import contexts

__all__ = ['CacheManager']


types.Basic.from_YASK = False
types.Array.from_YASK = True


class CacheManager(types.CacheManager):

    @classmethod
    def clear(cls):
        super(CacheManager, cls).clear()
        log("Dumping all contexts")
        contexts.dump()

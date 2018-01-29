from devito.logger import yask as log
import devito.types as types

from devito.yask import namespace
from devito.yask.wrappers import contexts

__all__ = ['CacheManager']


types.Basic.from_YASK = False
types.Basic.is_YaskGridObject = False
types.Array.from_YASK = True


class YaskGridObject(types.Object):

    is_YaskGridObject = True

    def __init__(self, mapped_function_name):
        self.mapped_function_name = mapped_function_name
        self.name = namespace['code-grid-name'](mapped_function_name)
        self.dtype = namespace['type-grid']
        self.value = None


class CacheManager(types.CacheManager):

    @classmethod
    def clear(cls):
        log("Dumping contexts and symbol caches")
        contexts.dump()
        super(CacheManager, cls).clear()

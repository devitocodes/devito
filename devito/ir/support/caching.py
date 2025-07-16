from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar


__all__ = ['CacheInstances', 'CacheInstancesMeta']


# Describes the type of a subclass of CacheInstances
InstanceType = TypeVar('InstanceType', bound='CacheInstances', covariant=True)


class CacheInstancesMeta(type):
    """
    Metaclass to wrap construction in an LRU cache.
    """

    _cache_clear_funcs: dict[type, Callable[[], None]] = {}

    def __init__(cls: type[InstanceType], *args) -> None:  # type: ignore
        super().__init__(*args)

        # Define an instance cache attribute and register a cache clear function
        cls._instance_cache: Callable | None = None
        CacheInstancesMeta._cache_clear_funcs[cls] = cls._clear_cache

    def __call__(cls: type[InstanceType],  # type: ignore
                 *args, **kwargs) -> InstanceType:
        if cls._instance_cache is None:
            maxsize = getattr(cls, '_instance_cache_size', 128)
            cls._instance_cache = lru_cache(maxsize=maxsize)(super().__call__)

        args, kwargs = cls._preprocess_args(*args, **kwargs)
        return cls._instance_cache(*args, **kwargs)

    @classmethod
    def clear_caches(cls: type['CacheInstancesMeta']) -> None:
        """
        Clear all caches for classes using this metaclass.
        """
        for clear_func in cls._cache_clear_funcs.values():
            clear_func()


class CacheInstances(metaclass=CacheInstancesMeta):
    """
    Parent class that wraps construction in an LRU cache.
    """

    @classmethod
    def _preprocess_args(cls, *args, **kwargs):
        """
        Preprocess the arguments before caching. This can be overridden in subclasses
        to customize argument handling (e.g. to convert to hashable types).
        """
        return args, kwargs

    @classmethod
    def _clear_cache(cls) -> None:
        """
        Clears the cache for this class, if any has been initialized.
        """
        if cls._instance_cache is not None:
            cls._instance_cache.cache_clear()  # type: ignore

    @staticmethod
    def clear_caches() -> None:
        """
        Clears all IR instance caches.
        """
        CacheInstancesMeta.clear_caches()

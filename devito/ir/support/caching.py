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

    _cached_types: set[type['CacheInstances']] = set()

    def __init__(cls: type[InstanceType], *args) -> None:  # type: ignore
        super().__init__(*args)

        # Register the cached type
        CacheInstancesMeta._cached_types.add(cls)

    def __call__(cls: type[InstanceType],  # type: ignore
                 *args, **kwargs) -> InstanceType:
        if cls._instance_cache is None:
            maxsize = cls._instance_cache_size
            cls._instance_cache = lru_cache(maxsize=maxsize)(super().__call__)

        args, kwargs = cls._preprocess_args(*args, **kwargs)
        return cls._instance_cache(*args, **kwargs)

    @classmethod
    def clear_caches(cls: type['CacheInstancesMeta']) -> None:
        """
        Clear all caches for classes using this metaclass.
        """
        for cached_type in cls._cached_types:
            if cached_type._instance_cache is not None:
                cached_type._instance_cache.cache_clear()


class CacheInstances(metaclass=CacheInstancesMeta):
    """
    Parent class that wraps construction in an LRU cache.
    """

    _instance_cache: Callable | None = None
    _instance_cache_size: int = 128

    @classmethod
    def _preprocess_args(cls, *args, **kwargs):
        """
        Preprocess the arguments before caching. This can be overridden in subclasses
        to customize argument handling (e.g. to convert to hashable types).
        """
        return args, kwargs

    @staticmethod
    def clear_caches() -> None:
        """
        Clears all IR instance caches.
        """
        CacheInstancesMeta.clear_caches()

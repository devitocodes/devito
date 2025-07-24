from collections.abc import Hashable, Iterator
from functools import lru_cache, partial, update_wrapper
from threading import RLock, local
from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar


__all__ = ['memoized_meth', 'memoized_generator', 'CacheInstances']


# Type variables for memoized method decorators
InstanceType = TypeVar('InstanceType', contravariant=True)
ParamsType = ParamSpec('ParamsType')
ReturnType = TypeVar('ReturnType', covariant=True)


class memoized_meth(Generic[InstanceType, ParamsType, ReturnType]):
    """
    Decorator for a cached instance method. There is one cache per thread stored
    on the object instance itself.
    """

    def __init__(self, meth: Callable[Concatenate[InstanceType, ParamsType],
                                      ReturnType]) -> None:
        self._meth = meth
        self._lock = RLock()  # Lock to safely initialize the thread-local object
        update_wrapper(self, self._meth)

    def __get__(self, obj: InstanceType, cls: type[InstanceType] | None = None) \
            -> Callable[ParamsType, ReturnType]:
        """
        Binds the memoized method to an instance.
        """
        return partial(self, obj)

    def _get_cache(self, obj: InstanceType) -> dict[Hashable, ReturnType]:
        """
        Retrieves the thread-local cache for the given object instance, initializing
        it if necessary.
        """
        # Try-catch is theoretically faster on the happy path
        _local: local
        try:
            # Attempt to access the thread-local data
            _local = obj._memoized_meth__local

        # If the cache doesn't exist, initialize it
        except AttributeError:
            with self._lock:
                # Check again in case another thread initialized outside the lock
                if not hasattr(obj, '_memoized_meth__local'):
                    # Initialize the local data if it doesn't exist
                    obj._memoized_meth__local = local()

            # Get the thread-local data
            _local = obj._memoized_meth__local

        # Local data is initialized; create or retrieve the cache
        try:
            return _local.cache
        except AttributeError:
            _local.cache = {}
            return _local.cache

    def __call__(self, obj: InstanceType,
                 *args: ParamsType.args, **kwargs: ParamsType.kwargs) -> ReturnType:
        """
        Invokes the memoized method, caching the result if it hasn't been evaluated yet.
        """
        # If arguments are not hashable, just evaluate the method directly
        if not isinstance(args, Hashable):
            return self._meth(obj, *args, **kwargs)

        # Get the local cache for the object instance
        cache = self._get_cache(obj)
        key = (self._meth, args, frozenset(kwargs.items()))
        try:
            # Try to retrieve the cached value
            res = cache[key]
        except KeyError:
            # If not cached, compute the value
            res = cache[key] = self._meth(obj, *args, **kwargs)

        return res


# Describes the type of element yielded by a cached iterator
YieldType = TypeVar('YieldType', covariant=True)


class SafeTee(Iterator[YieldType]):
    """
    A thread-safe version of `itertools.tee` that allows multiple iterators to safely
    share the same buffer.

    In theory, this comes at a cost to performance of iterating elements that haven't
    yet been generated, as `itertools.tee` is implemented in C (i.e. is fast) but we
    need to buffer (and lock) in Python instead.

    However, the lock is not needed for elements that have already been buffered,
    allowing for concurrent iteration after the generator is initially consumed.
    """
    def __init__(self, source_iter: Iterator[YieldType],
                 buffer: list[YieldType] = None, lock: RLock = None) \
            -> None:
        # If no buffer/lock are provided, this is a parent iterator
        self._source_iter = source_iter
        self._buffer = buffer if buffer is not None else []
        self._lock = lock if lock is not None else RLock()
        self._next = 0

    def __iter__(self) -> Iterator[YieldType]:
        return self

    def __next__(self) -> YieldType:
        """
        Safely retrieves the buffer if available, or generates the next element
        from the source iterator if not.
        """
        # Retry concurrent element access until we can return a value
        while True:
            if self._next < len(self._buffer):
                # If we have another buffered element, return it
                result = self._buffer[self._next]
                self._next += 1

                return result

            # Otherwise, we may need to generate a new element
            with self._lock:
                if self._next < len(self._buffer):
                    # Another thread has already generated the next element; retry
                    continue

                # Generate the next element from the source iterator
                try:
                    # Try to get the next element from the source iterator
                    result = next(self._source_iter)
                    self._buffer.append(result)
                    self._next += 1
                    return result
                except StopIteration:
                    # The source iterator has been exhausted
                    raise

    def __copy__(self) -> 'SafeTee':
        return SafeTee(self._source_iter, self._buffer, self._lock)

    def tee(self) -> Iterator[YieldType]:
        """
        Creates a new iterator that shares the same buffer and lock.
        """
        return self.__copy__()


class memoized_generator(Generic[InstanceType, ParamsType, YieldType]):
    """
    Decorator for a cached instance generator method. The initial call to the generator
    will block and return a thread-safe version of `itertools.tee` that allows for
    concurrent iteration.
    """

    def __init__(self, meth: Callable[Concatenate[InstanceType, ParamsType],
                                      Iterator[YieldType]]) -> None:
        self._meth = meth
        self._lock = RLock()  # Lock for initial generator calls
        update_wrapper(self, self._meth)

    def __get__(self, obj: InstanceType, cls: type[InstanceType] | None = None) \
            -> Callable[ParamsType, Iterator[YieldType]]:
        """
        Binds the memoized method to an instance.
        """
        return partial(self, obj)

    def _get_cache(self, obj: InstanceType) -> dict[Hashable, SafeTee[YieldType]]:
        """
        Retrieves the generator cache for the given object instance, initializing
        it if necessary.
        """
        # Try-catch is theoretically faster on the happy path
        try:
            # Attempt to access the cache directly
            return obj._memoized_generator__cache

        # If the cache doesn't exist, initialize it
        except AttributeError:
            with self._lock:
                # Check again in case another thread initialized outside the lock
                if not hasattr(obj, '_memoized_generator__cache'):
                    # Initialize the cache if it doesn't exist
                    obj._memoized_generator__cache = {}

            # Return the cache
            return obj._memoized_generator__cache

    def __call__(self, obj: InstanceType,
                 *args: ParamsType.args, **kwargs: ParamsType.kwargs) \
            -> Iterator[YieldType]:
        """
        Invokes the memoized generator, caching a SafeTee if it hasn't been created yet.
        """
        # If arguments are not hashable, just evaluate the method directly
        if not isinstance(args, Hashable):
            return self._meth(obj, *args, **kwargs)

        # Get the local cache for the object instance
        cache = self._get_cache(obj)
        key = (self._meth, args, frozenset(kwargs.items()))
        try:
            # Try to retrieve the cached value
            res = cache[key]
        except KeyError:
            # If not cached, compute the value
            source_iter = self._meth(obj, *args, **kwargs)
            res = cache[key] = SafeTee(source_iter)

        return res.tee()


# Describes the type of a subclass of CacheInstances
CachedInstanceType = TypeVar('CachedInstanceType',
                             bound='CacheInstances', covariant=True)


class CacheInstancesMeta(type):
    """
    Metaclass to wrap construction in an LRU cache.
    """

    _cached_types: set[type['CacheInstances']] = set()

    def __init__(cls: type[CachedInstanceType], *args) -> None:  # type: ignore
        super().__init__(*args)

        # Register the cached type
        CacheInstancesMeta._cached_types.add(cls)

    def __call__(cls: type[CachedInstanceType],  # type: ignore
                 *args, **kwargs) -> CachedInstanceType:
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

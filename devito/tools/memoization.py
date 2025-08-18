from collections.abc import Callable, Hashable
from functools import lru_cache, partial
from itertools import tee
from typing import TypeVar

__all__ = ['memoized_func', 'memoized_meth', 'memoized_generator', 'CacheInstances']


class memoized_func:
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated). This decorator may also be used on class methods,
    but it will cache at the class level; to cache at the instance level,
    use ``memoized_meth``.

    Adapted from: ::

        https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kw):
        if not isinstance(args, Hashable):
            # Uncacheable, a list, for instance.
            # Better to not cache than blow up.
            return self.func(*args, **kw)
        key = (self.func, args, frozenset(kw.items()))
        if key in self.cache:
            return self.cache[key]
        else:
            value = self.func(*args, **kw)
            self.cache[key] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return partial(self.__call__, obj)


class memoized_meth:
    """
    Decorator. Cache the return value of a class method.

    Unlike ``memoized_func``, the return value of a given method invocation
    will be cached on the instance whose method was invoked. All arguments
    passed to a method decorated with memoize must be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method: ::

        class Obj:
            @memoize
            def add_to(self, arg):
                return self + arg
        Obj.add_to(1) # not enough arguments
        Obj.add_to(1, 2) # returns 3, result is not cached

    Adapted from: ::

        code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        if not isinstance(args, Hashable):
            # Uncacheable, a list, for instance.
            # Better to not cache than blow up.
            return self.func(*args)
        obj = args[0]
        try:
            cache = obj.__cache_meth
        except AttributeError:
            cache = obj.__cache_meth = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


class memoized_generator:

    """
    Decorator. Cache the return value of an instance generator method.
    """

    def __init__(self, func):
        self.func = func

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kwargs):
        if not isinstance(args, Hashable):
            # Uncacheable, a list, for instance.
            # Better to not cache than blow up.
            return self.func(*args)
        obj = args[0]
        try:
            cache = obj.__cache_gen
        except AttributeError:
            cache = obj.__cache_gen = {}
        key = (self.func, args[1:], frozenset(kwargs.items()))
        it = cache[key] if key in cache else self.func(*args, **kwargs)
        cache[key], result = tee(it)
        return result


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

from collections.abc import Callable, Hashable
from functools import lru_cache, partial, wraps
from itertools import tee
from typing import TypeVar
from weakref import WeakKeyDictionary

__all__ = [
    'CacheInstances',
    'cached_hash',
    'memoized_func',
    'memoized_generator',
    'memoized_meth',
    'memoized_weak_meth',
    'reuse_if_unchanged'
]


def cached_hash(func):
    """
    Cache an immutable object's ``__hash__`` return value in ``_mhash``.

    Warning: avoid explicitly calling a superclass' cached ``__hash__`` on a
    subclass instance, as that would stash the superclass hash in ``_mhash``.

    Warning: avoid using it on pickled objects.
    """
    @wraps(func)
    def wrapper(self):
        try:
            return self._mhash
        except AttributeError:
            ret = func(self)
            self._mhash = ret
            return ret

    return wrapper


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

    # Long-lived caches for process-global helpers, such as arch discovery.
    _scope_persistent = 'persistent'
    # Build-scoped caches that may retain compiler inputs during Operator construction.
    _scope_build = 'build'
    _scoped_caches = {}

    def __new__(cls, func=None, *, scope=None):
        if func is None:
            return lambda f: cls(f, scope=scope)
        return super().__new__(cls)

    def __init__(self, func, *, scope=None):
        self.func = func
        self.scope = scope or self._scope_persistent
        self.cache = {}
        self._scoped_caches.setdefault(self.scope, set()).add(self)

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

    def clear(self):
        self.cache.clear()

    @classmethod
    def clear_scoped_caches(cls, scope):
        for cache in cls._scoped_caches.get(scope, ()):
            cache.clear()

    @classmethod
    def clear_build_caches(cls):
        cls.clear_scoped_caches(cls._scope_build)


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
        if kw:
            key = (self.func, args[1:], frozenset(kw.items()))
        else:
            key = (self.func, args[1:])

        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        except TypeError:
            # Uncacheable, e.g. an unhashable item within ``args``.
            return self.func(*args, **kw)

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


def memoized_weak_meth(*, key=None, freeze=None, thaw=None):
    """
    Cache a method result against its first argument using weak references.

    This is useful for visitors operating on temporary IR roots: the cache can
    be shared across short-lived visitor instances without keeping those roots
    alive. Only calls without extra arguments are cached; all other calls fall
    back to the wrapped method.

    Parameters
    ----------
    key : callable, optional
        A callable receiving ``self`` and returning a hashable cache partition.
    freeze : callable, optional
        Convert the method result before storing it in the cache.
    thaw : callable, optional
        Convert the cached value before returning it to the caller.
    """
    def decorator(func):
        caches = {}

        @wraps(func)
        def wrapper(self, o, *args, **kwargs):
            if args or kwargs:
                return func(self, o, *args, **kwargs)

            try:
                partition = key(self) if key is not None else None
                cache = caches.setdefault(partition, WeakKeyDictionary())
                ret = cache[o]
            except KeyError:
                ret = func(self, o)
                if freeze is not None:
                    ret = freeze(ret)
                cache[o] = ret
            except TypeError:
                return func(self, o)

            if thaw is not None:
                return thaw(ret)

            return ret

        return wrapper

    return decorator


# Describes the type of a subclass of CacheInstances
InstanceType = TypeVar('InstanceType', bound='CacheInstances', covariant=True)


class CacheInstancesMeta(type):
    """
    Metaclass to wrap construction in an LRU cache.
    """

    _cached_types: set[type['CacheInstances']] = set()

    def __init__(cls: type[InstanceType], *args) -> None:  # type: ignore
        super().__init__(*args)

        # Register the cached type and eagerly create its cache, bound to its
        # own constructor. Eager initialisation avoids a bug where a subclass
        # would inherit (and reuse) a parent's cache via MRO lookup if the
        # parent happened to be instantiated first.
        CacheInstancesMeta._cached_types.add(cls)
        maxsize = cls._instance_cache_size
        cls._instance_cache = lru_cache(maxsize=maxsize)(
            super().__call__
        )

    def __call__(cls: type[InstanceType],  # type: ignore
                 *args, **kwargs) -> InstanceType:
        if cls._instance_cache_size == 0:
            return super().__call__(*args, **kwargs)

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
    _instance_cache_size: int = 8192

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


def reuse_if_unchanged(fields):
    """
    Decorator for wrapper-style constructors that should return the original
    object when called as ``Cls(existing_obj, **same_metadata)``.

    The wrapped callable is assumed to be a classmethod-like constructor
    receiving ``cls`` as first argument. The fast path triggers only when:

    * the constructor is called with exactly one positional argument;
    * that argument is already an exact instance of ``cls``;
    * any explicitly provided metadata fields are the same objects as the
      corresponding attributes on the input object.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(cls, *args, **kwargs):
            if len(args) == 1:
                input_obj = args[0]
                if type(input_obj) is cls:
                    names = getattr(cls, fields) if isinstance(fields, str) else fields
                    for name in names:
                        if name in kwargs and \
                           kwargs[name] is not getattr(input_obj, name, None):
                            break
                    else:
                        return input_obj
            return func(cls, *args, **kwargs)

        return wrapper

    return decorator

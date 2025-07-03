from collections.abc import Hashable, Callable
from functools import partial, wraps
from itertools import tee
from typing import TypeVar

from devito.tools import WeakValueCache


__all__ = ['memoized_func', 'memoized_meth', 'memoized_generator']


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


# Describes the type of an object cached by `memoized_constructor`
InstanceType = TypeVar('InstanceType')
Constructor = Callable[..., InstanceType]


def _memoized_instances(cls: type[InstanceType]) -> type[InstanceType]:
    """
    Decorator for a class that caches instances based on the hash values of
    constructing arguments. The constructed values are stored weakly and
    evicted from the cache when no longer referenced.

    We need to override both __new__ and __init__ to ensure initialization
    only happens once for a cached instance.
    """

    # Check if we already decorated a parent class
    already_applied = getattr(cls, '_memoized_instances__exists', False)
    cls._memoized_instances__exists = True

    new = cls.__new__
    init = cls.__init__
    cache: WeakValueCache[InstanceType] = WeakValueCache(cls)

    @wraps(new)
    def _new(_cls: type[InstanceType], *args: Hashable,
             _memoized_instances__use_cache: bool = True,
             **kwargs: Hashable) -> InstanceType:
        # The decorator must be reapplied to a child class, so we make sure cls matches
        if _memoized_instances__use_cache and _cls is not cls:
            raise TypeError(f"_memoized_instances must be applied to {_cls.__name__}, "
                            f"not (just) {cls.__name__}")

        # The cache called the constructor; avoid infinite recursion
        if not _memoized_instances__use_cache:
            # If the class doesn't define __new__, we can't pass any args
            if new is object.__new__:
                obj = new(_cls)

            # Otherwise forward all arguments
            else:
                # If we applied the decorator to a parent class, forward the caching flag
                if already_applied:
                    kwargs['_memoized_instances__use_cache'] = False
                obj = new(_cls, *args, **kwargs)

            # Set our initialization flag and return
            obj._memoized_instances__initialized = False
            return obj

        return cache.get_or_create(*args, _memoized_instances__use_cache=False, **kwargs)

    @wraps(init)
    def _init(self: InstanceType, *args: Hashable, **kwargs: Hashable) -> None:
        # Skip reinitialization if this object was obtained from the cache
        try:
            if self._memoized_instances__initialized:
                return
        except AttributeError:
            # If the attribute doesn't exist, this is a new instance
            self._memoized_instances__initialized = False

        # Don't forward our extra argument to the original __init__
        kwargs.pop('_memoized_instances__use_cache', None)
        init(self, *args, **kwargs)
        self._memoized_instances__initialized = True

    def _copy(self: InstanceType) -> InstanceType:
        # Copy should just return the cached instance; bypass the cache machinery
        return self

    # Update the class's methods
    cls.__new__ = _new
    cls.__init__ = _init
    cls.__copy__ = _copy

    return cls

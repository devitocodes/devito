import gc
import weakref

import sympy

__all__ = ['Cached', '_SymbolCache', 'CacheManager']

_SymbolCache = {}
"""The symbol cache."""


class AugmentedWeakRef(weakref.ref):

    def __new__(cls, obj, meta):
        obj = super().__new__(cls, obj)
        obj.nbytes = meta.get('nbytes', 0)
        return obj


class Cached(object):

    """
    Mixin class for cached symbolic objects.
    """

    @classmethod
    def _cached(cls, key=None):
        """
        Test if a key is in the symbol cache and maps to an object that
        is still alive.

        Parameters
        ----------
        key : key, optional
            The cache key. If not supplied, use `cls._cache_key()`
        """
        key = key or cls._cache_key()
        if key not in _SymbolCache:
            return False
        return _SymbolCache[key]() is not None

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        """
        A unique, deterministic cache key from the input arguments.

        By default, the cache key is ``cls``, namely the class type itself.
        """
        return cls

    def __init__(self, key):
        """
        Store `self` in the symbol cache.

        Parameters
        ----------
        key : key
            The cache key.
        """
        # Precompute hash. This uniquely depends on the cache key
        self._hash_from_cachekey = hash(key)

        # Add ourselves to the symbol cache
        _SymbolCache[key] = AugmentedWeakRef(self, self._cache_meta())

    def __hash__(self):
        """
        The hash value of an object that caches on its type is the
        hash value of the type itself.
        """
        return self._hash_from_cachekey

    def _cache_meta(self):
        """
        Metadata attached when ``self`` is added to the symbol cache.

        Notes
        -----
        This should be specialized by the individual subclasses. This is useful
        to implement callbacks to be executed upon eviction.
        """
        return {}

    def _cached_init(self):
        """Initialise symbolic object with a cached object state."""
        original = _SymbolCache[self.__class__]
        self.__dict__ = original().__dict__


class CacheManager(object):

    """
    Drop unreferenced objects from the SymPy and Devito caches. The associated
    data is lost (and thus memory is freed).
    """

    gc_ths = 3*10**8
    """
    The `clear` function will trigger garbage collection if at least one weak
    reference points to an unreachable object whose size in bytes is greated
    than the `gc_ths` value. Garbage collection is an expensive operation, so
    we do it judiciously.
    """

    @classmethod
    def clear(cls, force=True):
        sympy.cache.clear_cache()

        # Maybe trigger garbage collection
        fire_gc = force or any(i.nbytes > cls.gc_ths for i in _SymbolCache.values())
        if fire_gc:
            gc.collect()

        for key, obj in list(_SymbolCache.items()):
            if obj() is None:
                del _SymbolCache[key]

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
    def _cache_key(cls, *args, **kwargs):
        """
        A unique, deterministic cache key from the input arguments.

        Notes
        -----
        To be implemented by subclasses.
        """
        raise NotImplementedError

    @classmethod
    def _cache_get(cls, key):
        """
        Retrieve the object corresponding to a given key. If the key is not in
        the symbol cache or if the mapped object is not alive anymore, returns None.

        Parameters
        ----------
        key : object
            The cache key. It must be hashable.
        """
        if key in _SymbolCache:
            # There is indeed an object mapped to `key`. But is it still alive?
            obj = _SymbolCache[key]()
            if obj is None:
                # Cleanup _SymbolCache (though practically unnecessary)
                del _SymbolCache[key]
                return None
            else:
                return obj
        else:
            return None

    def __init__(self, key):
        """
        Store `self` in the symbol cache.

        Parameters
        ----------
        key : object
            The cache key. It must be hashable.
        """
        # Precompute hash. This uniquely depends on the cache key
        self._cache_key_hash = hash(key)

        # Add ourselves to the symbol cache
        _SymbolCache[key] = AugmentedWeakRef(self, self._cache_meta())

    def __init_cached__(self, key):
        """
        Initialise `self` with a cached object state.

        Parameters
        ----------
        key : object
            The cache key of the object whose state is used to initialize `self`.
            It must be hashable.
        """
        self.__dict__ = _SymbolCache[key]().__dict__

    def __hash__(self):
        """
        The hash value of an object that caches on its type is the
        hash value of the type itself.
        """
        return self._cache_key_hash

    def _cache_meta(self):
        """
        Metadata attached when ``self`` is added to the symbol cache.

        Notes
        -----
        This should be specialized by the individual subclasses. This is useful
        to implement callbacks to be executed upon eviction.
        """
        return {}


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
        # Wipe out the "true" SymPy cache
        sympy.cache.clear_cache()

        # Wipe out the hidden module-private SymPy caches
        sympy.polys.rootoftools.ComplexRootOf.clear_cache()
        sympy.polys.rings._ring_cache.clear()
        sympy.polys.fields._field_cache.clear()
        sympy.polys.domains.modularinteger._modular_integer_cache.clear()

        # Maybe trigger garbage collection
        fire_gc = force or any(i.nbytes > cls.gc_ths for i in _SymbolCache.values())
        if fire_gc:
            gc.collect()

        for key, obj in list(_SymbolCache.items()):
            if obj() is None:
                del _SymbolCache[key]

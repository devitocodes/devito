import abc
import weakref
from concurrent.futures import Future
from hashlib import sha1
from threading import RLock
from typing import Generic, Hashable, TypeVar
from weakref import ReferenceType


__all__ = ['Tag', 'Signer', 'Reconstructable', 'Pickable', 'Singleton', 'Stamp',
           'WeakValueCache']


class Tag(abc.ABC):

    """
    An abstract class to define categories of object decorators.

    Notes
    -----
    This class must be subclassed for each new category.
    """

    def __init__(self, name, val=None):
        self.name = name
        self.val = val

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.val == other.val

    def __lt__(self, other):
        return self.val < other.val

    def __le__(self, other):
        return self.val <= other.val

    def __gt__(self, other):
        return self.val > other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __hash__(self):
        return hash((self.name, self.val))

    def __str__(self):
        if self.val is None:
            ret = self.name
        else:
            ret = "%s[%s]" % (self.name, str(self.val))
        return ret

    __repr__ = __str__


class Signer:

    """
    A base class for types that can emit a unique, deterministic
    string representing their internal state. Subclasses may be mutable or
    immutable.

    Notes
    -----
    Subclasses must implement the method :meth:`__signature_items___`.

    Regardless of whether an object is mutable or immutable, the returned
    signature must be immutable, and thus hashable.
    """

    @classmethod
    def _digest(cls, *signers):
        """Produce a unique, deterministic signature out of one or more
        ``signers`` objects."""
        items = []
        for i in signers:
            try:
                items.extend(list(i._signature_items()))
            except AttributeError:
                items.append(str(i))
        return cls._sign(items)

    @classmethod
    def _sign(cls, items):
        return sha1(''.join(items).encode()).hexdigest()

    def _signature_items(self):
        """Return a tuple of items from which the signature is computed. The
        items must be string. This method must be deterministic (i.e., the items
        must always be returned in the same order, even across different runs)."""
        return ()

    def _signature(self):
        return Signer._sign(self._signature_items())


class Reconstructable:

    __rargs__ = ()
    """
    The positional arguments to reconstruct the object.
    """

    __rkwargs__ = ()
    """
    The keyword arguments to reconstruct the object.
    """

    def _rebuild(self, *args, **kwargs):
        """
        Reconstruct `self` via `self.__class__(*args, **kwargs)` using
        `self`'s `__rargs__` and `__rkwargs__` if and where `*args` and
        `**kwargs` lack entries.

        Examples
        --------
        Given

            class Foo:
                __rargs__ = ('a', 'b')
                __rkwargs__ = ('c',)
                def __init__(self, a, b, c=4):
                    self.a = a
                    self.b = b
                    self.c = c

            a = foo(3, 5)`

        Then:

            * `a._rebuild() -> x(3, 5, 4)` (i.e., copy of `a`).
            * `a._rebuild(4) -> x(4, 5, 4)`
            * `a._rebuild(4, 7) -> x(4, 7, 4)`
            * `a._rebuild(c=5) -> x(3, 5, 5)`
            * `a._rebuild(1, c=7) -> x(1, 5, 7)`
        """
        for i in self.__rargs__[len(args):]:
            if i.startswith('*'):
                args += tuple(getattr(self, i[1:]))
            else:
                args += (getattr(self, i),)

        args = list(args)
        for k in list(kwargs):
            if k in self.__rargs__:
                args[self.__rargs__.index(k)] = kwargs.pop(k)

        kwargs.update({i: getattr(self, i) for i in self.__rkwargs__ if i not in kwargs})

        # If this object has SymPy assumptions associated with it, which were not
        # in the kwargs, then include them
        try:
            assumptions = self._assumptions_orig
            kwargs.update({k: v for k, v in assumptions.items() if k not in kwargs})
        except AttributeError:
            pass

        # Should we use a custom reconstructor?
        try:
            cls = self._rcls
        except AttributeError:
            cls = self.__class__

        return cls(*args, **kwargs)


class Pickable(Reconstructable):

    """
    A base class for types that require pickling. There are several complications
    that this class tries to handle: ::

        * Packages such as SymPy have their own way of handling pickling -- though
          still based upon Python's pickle module. This may get in conflict with
          other packages, or simply with Devito itself. For example, most of Devito
          symbolic objects are created via ``def __new__(..., **kwargs)``; SymPy1.1
          pickling does not cope nicely with ``new`` and ``kwargs``, since it is
          based on the low-level copy protocol (__reduce__, __reduce_ex__) and
          simply end up ignoring ``__getnewargs_ex__``, the function responsible
          for processing __new__'s kwargs.

    Notes
    -----
    All sub-classes using multiple inheritance may have to explicitly set
    ``__reduce_ex__ = Pickable.__reduce_ex__`` depending on the MRO.
    """

    @property
    def _pickle_rargs(self):
        """
        The positional arguments that need to be passed to __new__ upon unpickling.
        """
        # NOTE: Backward compatibility
        try:
            return self._pickle_args
        except AttributeError:
            pass

        return self.__rargs__

    @property
    def _pickle_rkwargs(self):
        """
        The keyword arguments that need to be passed to __new__ upon unpickling.
        """
        # NOTE: Backward compatibility
        try:
            return self._pickle_kwargs
        except AttributeError:
            pass

        return self.__rkwargs__

    @staticmethod
    def _pickle_wrapper(cls, args, kwargs):
        return cls.__new__(cls, *args, **kwargs)

    @property
    def _pickle_reconstructor(self):
        """
        Return the callable that should be used to reconstruct ``self`` upon
        unpickling. If None, default to whatever Python's pickle uses.
        """
        # NOTE: Backward compatibility
        try:
            return self._pickle_reconstruct
        except AttributeError:
            pass

        try:
            return self._rcls
        except AttributeError:
            return None

    def __reduce_ex__(self, proto):
        ret = object.__reduce_ex__(self, proto)
        reconstructor = self._pickle_reconstructor
        if reconstructor is None:
            return ret
        else:
            # Instead of the following wrapper function, we could use Python's copyreg
            _, (_, args, kwargs), state, iter0, iter1 = ret
            return (
                Pickable._pickle_wrapper,
                (reconstructor, args, kwargs),
                state,
                iter0,
                iter1,
            )

    def __getnewargs_ex__(self):
        args = []
        for i in self._pickle_rargs:
            if i.startswith('*'):
                args.extend(getattr(self, i[1:]))
            else:
                args.append(getattr(self, i))

        kwargs = {i: getattr(self, i) for i in self._pickle_rkwargs}

        return (tuple(args), kwargs)


class Singleton(type):

    """
    Metaclass for singleton classes.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Stamp:

    """
    Uniquely identify objects.
    """

    def __repr__(self):
        return "<%s>" % str(id(self))[-3:]

    __str__ = __repr__


# Cached instance type for `WeakValueCache` (not covariant to avoid weird init logic)
ValueType = TypeVar('ValueType')


class WeakValueCache(Generic[ValueType]):
    """
    A thread-safe cache that stores weak references to instances of a certain type,
    evicting entries when they are no longer reachable.

    Any thread querying with construction arguments that match an existing instance
    will receive the cached value; if another thread is currently constructing that
    value, the query thread will block until it's available. This ensures safe
    concurrent access while still allowing for threaded construction.
    """

    def __init__(self, cls: type[ValueType]):
        self._cls = cls
        self._futures: dict[int, Future[ReferenceType[ValueType]]] = {}
        self._lock = RLock()

    def _make_key(self, *args: Hashable, **kwargs: Hashable) -> int:
        return hash((*args, frozenset(kwargs.items())))

    def _create_instance(self, *args: Hashable, **kwargs: Hashable) -> ValueType:
        if self._cls is object.__new__:
            # If the constructor is object's __new__, we cannot pass any arguments
            obj = self._cls()
        else:
            # Otherwise, forward all construction arguments
            obj = self._cls(*args, **kwargs)

        # Initialize the object so it's ready for consuming threads
        obj.__init__(*args, **kwargs)

        return obj

    def get_or_create(self, *args: Hashable, **kwargs: Hashable) -> ValueType:
        """
        Gets an instance for the given construction arguments, creating it on this thread
        if it doesn't exist. If another thread is currently constructing it, blocks until
        the instance is available.
        """
        key = self._make_key(*args, **kwargs)
        future = self._futures.get(key, None)
        if future is not None:
            # Block until the object is available
            obj_ref = future.result()
            obj = obj_ref()

            if obj is None:
                # The object was garbage collected but future has yet to be cleared
                return self.get_or_create(*args, **kwargs)  # Retry to create

            # The object is available
            return obj

        # Don't use a context manager; we need to release before the recursive call
        self._lock.acquire()

        # Check that another thread hasn't created the future while we spun
        future = self._futures.get(key, None)
        if future is not None:
            # Release the lock and retry to retrieve the existing future
            self._lock.release()
            return self.get_or_create(*args, **kwargs)

        # This thread will supply the value
        future: Future[ReferenceType[ValueType]] = Future()
        self._futures[key] = future

        # Release the lock to allow for concurrent construction
        self._lock.release()

        # Perform construction outside the lock to avoid blocking other threads
        try:
            obj = self._create_instance(*args, **kwargs)

            # Listener for when the weak reference expires
            def on_obj_destroyed(k: int = key,
                                 f: Future[ReferenceType[ValueType]] = future) \
                    -> None:
                with self._lock:
                    if self._futures.get(k, None) is f:
                        del self._futures[k]

            # Register the callback and store a weak reference in the new future
            weakref.finalize(obj, on_obj_destroyed)
            future.set_result(weakref.ref(obj))

            return obj

        except Exception as e:
            # If the supplier failed, clean up the future and re-raise
            with self._lock:
                if self._futures.get(key) is future:
                    del self._futures[key]

            future.set_exception(e)
            raise e from None

    def clear(self):
        """
        Clears all entries in the cache.

        Objects currently being constructed will still be returned to callers waiting for
        them, but they will not be retrievable after this call.
        """
        with self._lock:
            self._futures.clear()

    def __len__(self) -> int:
        """
        Returns the number of keys currently in the cache.
        """
        with self._lock:
            return len(self._futures)

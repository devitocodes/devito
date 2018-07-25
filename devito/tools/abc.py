from abc import ABC
from hashlib import sha1


__all__ = ['Tag', 'Signer', 'Pickable']


class Tag(ABC):

    """
    An abstract class to define categories of object decorators.

    .. note::

        This class must be subclassed for each new category.
    """

    def __init__(self, name, val=None):
        self.name = name
        self.val = val

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.val == other.val

    def __hash__(self):
        return hash((self.name, self.val))

    def __str__(self):
        if self.val is None:
            ret = self.name
        else:
            ret = "%s[%s]" % (self.name, str(self.val))
        return ret

    __repr__ = __str__


class Signer(object):

    """
    A base class for types that can emit a unique, deterministic
    string representing their internal state. Subclasses may be mutable or
    immutable.

    .. note::

        Subclasses must implement the method :meth:`__signature_items___`.

    .. note::

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


class Pickable(object):

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

    .. note::

        All sub-classes using multiple inheritance may have to explicitly set
        ``__reduce_ex__ = Pickable.__reduce_ex__`` depending on the MRO.
    """

    _pickle_args = []
    """The positional arguments that need to be passed to __new__ upon unpickling."""

    _pickle_kwargs = []
    """The keyword arguments that need to be passed to __new__ upon unpickling."""

    @property
    def _pickle_reconstruct(self):
        """
        Return the callable that should be used to reconstruct ``self`` upon
        unpickling. If None, default to whatever Python's pickle uses.
        """
        return None

    def __reduce_ex__(self, proto):
        ret = object.__reduce_ex__(self, proto)
        reconstructor = self._pickle_reconstruct
        if reconstructor is None:
            return ret
        else:
            # Instead of the following wrapper function, we could use Python's copyreg
            def wrapper(cls, args, kwargs):
                return cls.__new__(cls, *args, **kwargs)
            _, (_, args, kwargs), state, iter0, iter1 = ret
            return (wrapper, (reconstructor, args, kwargs), state, iter0, iter1)

    def __getnewargs_ex__(self):
        return (tuple(getattr(self, i) for i in self._pickle_args),
                {i: getattr(self, i) for i in self._pickle_kwargs})

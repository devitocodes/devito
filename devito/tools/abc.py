from abc import ABC
from hashlib import sha1


__all__ = ['Tag', 'Signer']


class Tag(ABC):

    """
    An abstract class to define categories of object decorators.

    .. note::

        This class must be subclassed for each new category.
    """

    _repr = 'AbstractTag'

    _KNOWN = []

    def __init__(self, name, val=None):
        self.name = name
        self.val = val

        self._KNOWN.append(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.val == other.val

    def __hash__(self):
        return hash((self.name, self.val))

    def __str__(self):
        return self.name if self.val is None else '%s%s' % (self.name, str(self.val))

    def __repr__(self):
        if self.val is None:
            return "%s: %s" % (self._repr, self.name)
        else:
            return "%s: %s[%s]" % (self._repr, self.name, str(self.val))


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

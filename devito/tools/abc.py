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


class Signer(ABC):

    """
    An abstract class to represent types that can emit a unique, deterministic
    string representing their internal state. Subclasses may be mutable or
    immutable.

    .. note::

        Subclasses must implement the method :meth:`__signature__`.

    .. note::

        Regardless of whether an object is mutable or immutable, the returned
        signature must be immutable, and thus hashable.
    """

    @classmethod
    def digest(cls, *signable):
        """Produce a unique, deterministic signature out of one or more
        ``signable`` objects."""
        signatures = [i.__signature__() for i in signable]
        if len(signatures) == 1:
            return signatures[0]
        else:
            return sha1(''.join(signatures).encode()).hexdigest()

    def __signature__(self):
        raise NotImplementedError('Subclasses must implement `__signature__`')

from hashlib import sha1

__all__ = ['Signer']


class Signer(object):

    """
    A mixin for classes that can emit a unique, deterministic string
    representing their state. Subclasses may be mutable or immutable; if
    immutable, the signature depends on the internal state of the object.

    .. note::

        Subclasses must implement the method :meth:`__signature__`.
    """

    def __signature__(self):
        raise NotImplementedError('Subclasses must implement `sign`')

    @classmethod
    def digest(cls, signable):
        """Produce a unique, deterministic signature out of one or more
        ``signable`` objects."""
        return sha1(''.join(as_tuple(signable)).encode()).hexdigest()

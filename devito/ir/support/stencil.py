from collections import namedtuple

from devito.tools import DefaultOrderedDict, as_tuple

__all__ = ['Stencil']


class Stencil(DefaultOrderedDict):

    """
    A Stencil is a mapping from :class:`Dimension` symbols to the set of integer
    offsets used with it in expressions (the "neighboring points accessed").

    This also include zero offsets.

    The mapping is ordered based on the order in which dimensions are encountered
    (if extracted from expressions) or inserted.

    Note: Expressions must have been indexified for a Stencil to be computed.
    """

    def __init__(self, entries=None):
        """
        Initialize the Stencil.

        :param entries: An iterable of :class:`StencilEntry` or a 2-tuple
                        convertible into a :class:`StencilEntry` (i.e., a
                        :class:`Dimension` and a set).
        """
        processed = []
        for i in (entries or []):
            if isinstance(i, StencilEntry):
                processed.append((i.dim, i.ofs))
            elif isinstance(i, tuple):
                entry = StencilEntry(*i)  # Implicit type check
                processed.append((entry.dim, set(as_tuple(entry.ofs))))
            else:
                raise TypeError('Cannot construct a Stencil for %s' % str(i))
        super(Stencil, self).__init__(set, processed)

    @classmethod
    def union(cls, *dicts):
        """
        Compute the union of an iterable of :class:`Stencil` objects.
        """
        output = Stencil()
        for i in dicts:
            for k, v in i.items():
                output[k] |= v
        return output

    @property
    def empty(self):
        return all(len(i) == 0 for i in self.values())

    @property
    def dimensions(self):
        return list(self.keys())

    @property
    def entries(self):
        return tuple(StencilEntry(k, frozenset(v)) for k, v in self.items())

    @property
    def diameter(self):
        return {k: abs(max(v) - min(v)) for k, v in self.items()}

    def subtract(self, o):
        """
        Compute the set difference of each Dimension in self with the corresponding
        Dimension in ``o``.
        """
        output = Stencil()
        for k, v in self.items():
            output[k] = set(v)
            if k in o:
                output[k] -= o[k]
        return output

    def add(self, o):
        """
        Compute the set union of each Dimension in self with the corresponding
        Dimension in ``o``.
        """
        output = Stencil()
        for k, v in self.items():
            output[k] = set(v)
            if k in o:
                output[k] |= o[k]
        return output

    def get(self, k, v=None):
        obj = super(Stencil, self).get(k, v)
        return frozenset([0]) if obj is None else obj

    def entry(self, k):
        return StencilEntry(k, self.get(k))

    def prefix(self, o):
        """
        Return the common prefix of ``self`` and ``o`` as a new Stencil.
        """
        output = Stencil()
        for (k1, v1), (k2, v2) in zip(self.items(), o.items()):
            if k1 == k2 and v1 == v2:
                output[k1] = set(v1)
            else:
                break
        return output

    def copy(self):
        """
        Return a deep copy of the Stencil.
        """
        return Stencil(self.entries)

    def __eq__(self, other):
        return self.entries == other.entries

    def __ne__(self, other):
        return not self.__eq__(other)

    def __setitem__(self, key, val):
        entry = StencilEntry(key, val)  # Type checking
        super(Stencil, self).__setitem__(entry.dim, entry.ofs)


StencilEntry = namedtuple('StencilEntry', 'dim ofs')
StencilEntry.copy = lambda i: StencilEntry(i.dim, set(i.ofs))

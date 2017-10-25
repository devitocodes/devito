from sympy import Basic, Eq

from devito.symbolics import retrieve_indexed
from devito.tools import as_tuple, is_integer, flatten
from devito.types import Indexed


class Vector(tuple):

    """
    A representation of a vector in Z^n.

    The elements of a Vector can be integers or any SymPy expression.

    Notes on Vector comparison
    ==========================

    # Vector-scalar comparison
    --------------------------
    If a comparison between a vector and a non-vector is attempted, then the
    non-vector is promoted to a vector; if this is not possible, an exception
    is raised. This is handy because it turns a vector-scalar comparison into
    a vector-vector comparison with the scalar broadcasted to all vector entries.
    For example: ::

        (3, 4, 5) > 4 => (3, 4, 5) > (4, 4, 4) => False

    # Comparing Vector entries when these are SymPy expression
    ----------------------------------------------------------
    When we compare two entries that are both generic SymPy expressions, it is
    generally not possible to determine the truth value of the relation. For
    example, the truth value of `3*i < 4*j` cannot be determined. In some cases,
    however, the comparison is feasible; for example, `i + 4 < i` should always
    return false. A sufficient condition for two Vectors to be comparable is that
    their pair-wise indices are affine functions of the same variables, with
    coefficient 1.
    """

    def __new__(cls, *items):
        if not all(is_integer(i) or isinstance(i, Basic) for i in items):
            raise TypeError("Illegal Vector element type")
        return super(Vector, cls).__new__(cls, items)

    def _asvector(func):
        def wrapper(self, other):
            if not isinstance(other, Vector):
                try:
                    other = Vector(*other)
                except TypeError:
                    # Not iterable
                    other = Vector(*(as_tuple(other)*len(self)))
            if len(self) != len(other):
                raise TypeError("Cannot operate with Vectors of different rank")
            return func(self, other)
        return wrapper

    @_asvector
    def __add__(self, other):
        return Vector(*[i + j for i, j in zip(self, other)])

    @_asvector
    def __radd__(self, other):
        return self + other

    @_asvector
    def __sub__(self, other):
        return Vector(*[i - j for i, j in zip(self, other)])

    @_asvector
    def __rsub__(self, other):
        return self - other

    @_asvector
    def __eq__(self, other):
        return super(Vector, self).__eq__(other)

    @_asvector
    def __ne__(self, other):
        return super(Vector, self).__ne__(other)

    @_asvector
    def __lt__(self, other):
        try:
            diff = [int(i) for i in self.order(other)]
        except TypeError:
            raise TypeError("Cannot compare due to non-comparable index functions")
        return diff < [0]*self.rank

    @_asvector
    def __gt__(self, other):
        return other.__lt__(self)

    @_asvector
    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    @_asvector
    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __getitem__(self, key):
        ret = super(Vector, self).__getitem__(key)
        return Vector(*ret) if isinstance(key, slice) else ret

    def __repr__(self):
        maxlen = max(3, max([len(str(i)) for i in self]))
        return '\n'.join([('|{:^%d}|' % maxlen).format(str(i)) for i in self])

    @property
    def rank(self):
        return len(self)

    @property
    def sum(self):
        return sum(self)

    def distance(self, other):
        """Compute vector distance from ``self`` to ``other``."""
        return self - other

    def order(self, other):
        """
        A reflexive, transitive, and anti-symmetric relation for total ordering.

        Return a tuple of length equal to the Vector ``rank``. The i-th tuple
        entry, of type int, indicates whether the i-th component of ``self``
        precedes (< 0), equals (== 0), or succeeds (> 0) the i-th component of
        ``other``.
        """
        return self.distance(other)


class IterationInstance(Vector):

    """A representation of the iteration space point accessed by a
    :class:`Indexed` object."""

    def __new__(cls, indexed):
        assert isinstance(indexed, Indexed)
        obj = super(IterationInstance, cls).__new__(cls, *indexed.indices)
        obj.findices = tuple(indexed.base.function.indices)
        return obj

    def __eq__(self, other):
        if isinstance(other, IterationInstance) and self.findices != other.findices:
            raise TypeError("Cannot compare due to mismatching `findices`")
        return super(IterationInstance, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, IterationInstance) and self.findices != other.findices:
            raise TypeError("Cannot compare due to mismatching `findices`")
        return super(IterationInstance, self).__lt__(other)

    def __gt__(self, other):
        return other.__lt__(self)

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __repr__(self):
        return "IS([%s])" % ', '.join(str(i) for i in self)

    def distance(self, sink, dim=None):
        """Compute vector distance from ``self`` to ``sink``. If ``dim`` is
        supplied, compute the vector distance up to and including ``dim``."""
        if not isinstance(sink, IterationInstance):
            raise TypeError("Cannot compute distance from obj of type %s", type(sink))
        if self.findices != sink.findices:
            raise TypeError("Cannot compute distance due to mismatching `findices`")
        if dim is not None:
            try:
                limit = self.findices.index(dim) + 1
            except ValueError:
                raise TypeError("Cannot compute distance as `dim` is not in `findices`")
        else:
            limit = self.rank
        return super(IterationInstance, self).distance(sink)[:limit]

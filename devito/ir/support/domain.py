import abc
from collections import OrderedDict

from devito.tools import as_tuple

__all__ = ['NullInterval', 'Interval', 'DataSpace', 'IterationSpace']


class AbstractInterval(object):

    """
    A representation of a closed interval on Z.
    """

    __metaclass__ = abc.ABCMeta

    is_Null = False
    is_Defined = False

    def __init__(self, dim):
        self.dim = dim

    @classmethod
    def _apply_op(cls, intervals, key):
        """
        Return a new :class:`Interval` resulting from the iterative application
        of the method ``key`` over the :class:`Interval`s in ``intervals``, i.e.:
        ``intervals[0].key(intervals[1]).key(intervals[2])...``.
        """
        intervals = as_tuple(intervals)
        partial = intervals[0]
        for i in intervals[1:]:
            partial = getattr(partial, key)(i)
        return partial

    @abc.abstractmethod
    def _rebuild(self):
        return

    def intersection(self, o):
        return self._rebuild()

    @abc.abstractmethod
    def union(self, o):
        return self._rebuild()

    def subtract(self, o):
        return self._rebuild()

    def negate(self):
        return self._rebuild()

    @abc.abstractmethod
    def overlap(self, o):
        return

    def __eq__(self, o):
        return type(self) == type(o) and self.dim == o.dim

    def __hash__(self):
        return hash(self.dim.name)


class NullInterval(AbstractInterval):

    is_Null = True

    def __repr__(self):
        return "%s[Null]" % self.dim

    def _rebuild(self):
        return NullInterval(self.dim)

    def union(self, o):
        if self.dim == o.dim:
            return o._rebuild()
        else:
            return Space([self._rebuild(), o._rebuild()])

    def overlap(self, o):
        return False


class Interval(AbstractInterval):

    """
    Interval(dim, lower, upper)

    Create an :class:`Interval` of extent: ::

        dim.extent + abs(upper - lower)
    """

    is_Defined = True

    def __init__(self, dim, lower, upper):
        assert isinstance(lower, int)
        assert isinstance(upper, int)
        super(Interval, self).__init__(dim)
        self.lower = lower
        self.upper = upper
        self.min_extent = abs(upper - lower)
        self.extent = dim.symbolic_size + self.min_extent

    def __repr__(self):
        return "%s[%s, %s]" % (self.dim, self.lower, self.upper)

    def _rebuild(self):
        return Interval(self.dim, self.lower, self.upper)

    @property
    def limits(self):
        return (self.lower, self.upper)

    def intersection(self, o):
        if self.overlap(o):
            return Interval(self.dim, max(self.lower, o.lower), min(self.upper, o.upper))
        else:
            return NullInterval(self.dim)

    def union(self, o):
        if self.overlap(o):
            return Interval(self.dim, min(self.lower, o.lower), max(self.upper, o.upper))
        elif o.is_Null and self.dim == o.dim:
            return self._rebuild()
        else:
            return Space([self._rebuild(), o._rebuild()])

    def subtract(self, o):
        if self.dim != o.dim or o.is_Null:
            return self._rebuild()
        else:
            return Interval(self.dim, self.lower - o.lower, self.upper - o.upper)

    def negate(self):
        return Interval(self.dim, -self.lower, -self.upper)

    def overlap(self, o):
        if self.dim != o.dim:
            return False
        try:
            # In the "worst case scenario" the dimension extent is 0
            # so we can just neglect it
            min_extent = max(self.min_extent, o.min_extent)
            return (self.lower <= o.lower and o.lower <= self.lower + min_extent) or\
                (self.lower >= o.lower and self.lower <= o.lower + min_extent)
        except AttributeError:
            return False

    def __eq__(self, o):
        return super(Interval, self).__eq__(o) and\
            self.lower == o.lower and self.upper == o.upper

    def __hash__(self):
        return hash((self.dim.name, self.lower, self.upper))


class Space(object):

    """
    A bag of :class:`Interval`s.

    The intervals input ordering is honored.
    """

    def __init__(self, intervals):
        self.intervals = as_tuple(intervals)

    def __repr__(self):
        return "%s[%s]" % (self.__class__.__name__,
                           ', '.join([repr(i) for i in self.intervals]))

    def __eq__(self, o):
        return set(self.intervals) == set(o.intervals)

    def _construct(self, intervals):
        return Space(intervals)

    @property
    def size(self):
        return len(self.intervals)

    @property
    def empty(self):
        return self.size == 0

    @property
    def is_well_defined(self):
        """
        Return True if the space :class:`Interval`s are over different
        :class:`Dimension`s, False otherwise.
        """
        dims = [i.dim for i in self.intervals]
        return len(dims) == len(set(dims))

    @classmethod
    def generate(self, op, *spaces):
        """
        generate(op, *spaces)

        Create a new :class:`Space` from the iterative application of the
        operation ``op`` to the :class:`Space`s in ``spaces``.

        :param op: Any legal :class:`Interval` operation, such as ``intersection``
                   or ``union``. This should be provided as a string.
        :param spaces: An iterable of :class:`Space`s.

        Example
        -------
        space0 = Space([Interval(x, 1, -1)])
        space1 = Space([Interval(x, 2, -2), Interval(y, 3, -3)])
        space2 = Space([Interval(y, 2, -2), Interval(z, 1, -1)])

        res = Space.generate('intersection', space0, space1, space2)
        res --> Space([Interval(x, 2, -2), Interval(y, 3, -3), Interval(z, 1, -1)])
        """
        mapper = OrderedDict()
        for i in spaces:
            for interval in i.intervals:
                mapper.setdefault(interval.dim, []).append(interval)
        return Space([Interval._apply_op(v, op) for v in mapper.values()])

    def intersection(self, *spaces):
        mapper = OrderedDict([(i.dim, [i]) for i in self.intervals])
        for i in spaces:
            for interval in i.intervals:
                mapper.get(interval.dim, []).append(interval)
        return self._construct([Interval._apply_op(v, 'intersection')
                                for v in mapper.values()])

    def subtract(self, o):
        mapper = OrderedDict([(i.dim, i) for i in o.intervals])
        intervals = [i.subtract(mapper.get(i.dim, NullInterval(i.dim)))
                     for i in self.intervals]
        return self._construct(intervals)

    def drop(self, d):
        return self._construct([i._rebuild() for i in self.intervals
                                if i.dim not in as_tuple(d)])

    def negate(self):
        return self._construct([i.negate() for i in self.intervals])

    def __getitem__(self, dim):
        if not self.is_well_defined:
            raise ValueError("Cannot fetch Interval from ill defined Space")
        for i in self.intervals:
            if i.dim == dim:
                return i


class DataSpace(Space):
    pass


class IterationSpace(Space):

    """
    A :class:`Space` associating one or more :class:`Dimension`s to a
    :class:`Interval`. The interval is the data space, whereas the dimensions
    are the objects used to traverse the data space.
    """

    def __init__(self, intervals, sub_iterators):
        super(IterationSpace, self).__init__(intervals)
        self.sub_iterators = sub_iterators

    def _construct(self, intervals):
        return IterationSpace(intervals, self.sub_iterators)

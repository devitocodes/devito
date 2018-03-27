import abc
from collections import OrderedDict

from frozendict import frozendict

from devito.tools import as_tuple, filter_ordered

__all__ = ['NullInterval', 'Interval', 'IntervalGroup', 'IterationSpace', 'DataSpace',
           'Forward', 'Backward', 'Any']


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

    merge = union

    def subtract(self, o):
        return self._rebuild()

    def negate(self):
        return self._rebuild()

    zero = negate
    flip = negate

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

    def __hash__(self):
        return hash(self.dim)

    def _rebuild(self):
        return NullInterval(self.dim)

    def union(self, o):
        if self.dim == o.dim:
            return o._rebuild()
        else:
            return IntervalGroup([self._rebuild(), o._rebuild()])

    merge = union

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

    def __hash__(self):
        return hash((self.dim, self.limits))

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
            return IntervalGroup([self._rebuild(), o._rebuild()])

    def merge(self, o):
        if self.dim != o.dim or o.is_Null:
            return self._rebuild()
        else:
            return Interval(self.dim, min(self.lower, o.lower), max(self.upper, o.upper))

    def subtract(self, o):
        if self.dim != o.dim or o.is_Null:
            return self._rebuild()
        else:
            return Interval(self.dim, self.lower - o.lower, self.upper - o.upper)

    def negate(self):
        return Interval(self.dim, -self.lower, -self.upper)

    def zero(self):
        return Interval(self.dim, 0, 0)

    def flip(self):
        return Interval(self.dim, self.upper, self.lower)

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


class IntervalGroup(tuple):

    """
    A sequence of :class:`Interval`s with set-like operations exposed.
    """

    def __eq__(self, o):
        return set(self) == set(o)

    def __repr__(self):
        return "IntervalGroup[%s]" % (', '.join([repr(i) for i in self]))

    def __hash__(self):
        return hash(i for i in self)

    @property
    def dimensions(self):
        return filter_ordered([i.dim for i in self])

    @property
    def is_well_defined(self):
        """
        Return True if all :class:`Interval`s are over different :class:`Dimension`s,
        False otherwise.
        """
        return len(self.dimensions) == len(set(self.dimensions))

    @classmethod
    def generate(self, op, *interval_groups):
        """
        generate(op, *interval_groups)

        Create a new :class:`IntervalGroup` from the iterative application of the
        operation ``op`` to the :class:`IntervalGroup`s in ``interval_groups``.

        :param op: Any legal :class:`Interval` operation, such as ``intersection``
                   or ``union``. This should be provided as a string.
        :param interval_groups: An iterable of :class:`IntervalGroup`s.

        Example
        -------
        ig0 = IntervalGroup([Interval(x, 1, -1)])
        ig1 = IntervalGroup([Interval(x, 2, -2), Interval(y, 3, -3)])
        ig2 = IntervalGroup([Interval(y, 2, -2), Interval(z, 1, -1)])

        ret = IntervalGroup.generate('intersection', ig0, ig1, ig2)
        ret -> IntervalGroup([Interval(x, 2, -2), Interval(y, 3, -3), Interval(z, 1, -1)])
        """
        mapper = OrderedDict()
        for ig in interval_groups:
            for i in ig:
                mapper.setdefault(i.dim, []).append(i)
        return IntervalGroup(Interval._apply_op(v, op) for v in mapper.values())

    def intersection(self, o):
        mapper = OrderedDict([(i.dim, i) for i in o])
        intervals = [i.intersection(mapper.get(i.dim, i)) for i in self]
        return IntervalGroup(intervals)

    def subtract(self, o):
        mapper = OrderedDict([(i.dim, i) for i in o])
        intervals = [i.subtract(mapper.get(i.dim, NullInterval(i.dim))) for i in self]
        return IntervalGroup(intervals)

    def drop(self, d):
        return IntervalGroup([i._rebuild() for i in self if i.dim not in as_tuple(d)])

    def negate(self):
        return IntervalGroup([i.negate() for i in self])

    def zero(self, d=None):
        d = self.dimensions if d is None else as_tuple(d)
        return IntervalGroup([i.zero() if i.dim in d else i for i in self])

    def __getitem__(self, key):
        if isinstance(key, (slice, int)):
            return super(IntervalGroup, self).__getitem__(key)
        if not self.is_well_defined:
            raise ValueError("Cannot fetch Interval from ill defined Space")
        for i in self:
            if i.dim == key:
                return i
        return NullInterval(key)


class IterationDirection(object):

    """
    A representation of the direction in which an iteration space is traversed.
    """

    def __init__(self, name):
        self._name = name

    def __eq__(self, other):
        return isinstance(other, IterationDirection) and self._name == other._name

    def __repr__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)


Forward = IterationDirection('++')
"""Forward iteration direction ('++')."""

Backward = IterationDirection('--')
"""Backward iteration direction ('--')."""

Any = IterationDirection('*')
"""Wildcard direction (both '++' and '--' would be OK)."""


class IterationInterval(object):

    """
    An :class:`Interval` associated with an :class:`IterationDirection`.
    """

    def __init__(self, interval, direction):
        self.interval = interval
        self.direction = direction

    def __repr__(self):
        return "%s%s" % (self.interval, self.direction)

    def __eq__(self, other):
        return isinstance(other, IterationInterval) and\
            self.interval == other.interval and self.direction == other.direction

    def __hash__(self):
        return hash((self.interval, self.direction))

    @property
    def dim(self):
        return self.interval.dim

    @property
    def limits(self):
        return self.interval.limits


class Space(object):

    """
    A representation of a compact N-dimensional space as a sequence of
    :class:`Interval`s along N :class:`Dimension`s.

    :param intervals: A sequence of :class:`Interval`s describing the space.
    """

    def __init__(self, intervals):
        self._intervals = IntervalGroup(as_tuple(intervals))

    def __repr__(self):
        return "%s[%s]" % (self.__class__.__name__,
                           ", ".join(repr(i) for i in self.intervals))

    def __eq__(self, other):
        return isinstance(other, Space) and self.intervals == other.intervals

    def __hash__(self):
        return hash(self.intervals)

    @property
    def intervals(self):
        return self._intervals

    @property
    def size(self):
        return len(self.intervals)

    @property
    def empty(self):
        """Return True if this space has no intervals (no matter whether they
        are defined or null intervals), False otherwise."""
        return self.size == 0

    @property
    def dimensions(self):
        return filter_ordered(self.intervals.dimensions)


class DataSpace(Space):

    """
    A representation of a data space.

    :param intervals: A sequence of :class:`Interval`s describing the data space.
    :param parts: A mapper from :class:`Function`s to iterables of :class:`Interval`
                  describing the individual components of the data space.
    """

    def __init__(self, intervals, parts):
        super(DataSpace, self).__init__(intervals)
        self._parts = frozendict(parts)

    def __eq__(self, other):
        return isinstance(other, DataSpace) and\
            self.intervals == other.intervals and self.parts == other.parts

    def __hash__(self):
        return hash((super(DataSpace, self).__hash__(), self.parts))

    @classmethod
    def merge(cls, *others):
        if not others:
            return DataSpace(IntervalGroup(), {})
        intervals = IntervalGroup.generate('merge', *[i.intervals for i in others])
        parts = {}
        for i in others:
            for k, v in i.parts.items():
                parts.setdefault(k, []).append(v)
        parts = {k: IntervalGroup.generate('merge', *v) for k, v in parts.items()}
        return DataSpace(intervals, parts)

    @property
    def parts(self):
        return self._parts

    def __getitem__(self, key):
        ret = self.intervals[key]
        if ret.is_Null:
            try:
                ret = self._parts[key]
            except KeyError:
                ret = IntervalGroup()
        return ret

    def zero(self, d=None):
        intervals = self.intervals.zero(d)
        parts = {k: v.zero(d) for k, v in self.parts.items()}
        return DataSpace(intervals, parts)


class IterationSpace(Space):

    """
    A representation of an iteration space and its traversal through
    :class:`Interval`s and :class:`IterationDirection`s. For each interval,
    an arbitrary number of (sub-)iterators may be specified (see below).

    :param intervals: An ordered sequence of :class:`Interval`s defining the
                      iteration space.
    :param sub_iterators: (Optional) a mapper from :class:`Dimension`s in
                          ``intervals`` to iterables of :class:`DerivedDimension`,
                          which represent sub-dimensions along which the iteration
                          space is traversed.
    :param directions: (Optional) a mapper from :class:`Dimension`s in
                       ``intervals`` to :class:`IterationDirection`s.
    """

    def __init__(self, intervals, sub_iterators=None, directions=None):
        super(IterationSpace, self).__init__(intervals)
        self._sub_iterators = frozendict(sub_iterators or {})
        self._directions = frozendict(directions or {})

    def __repr__(self):
        ret = ', '.join(["%s%s" % (repr(i), repr(self.directions[i.dim]))
                         for i in self.intervals])
        return "IterationSpace[%s]" % ret

    def __eq__(self, other):
        return isinstance(other, IterationSpace) and\
            self.intervals == other.intervals and self.directions == other.directions

    def __hash__(self):
        return hash((super(IterationSpace, self).__hash__(), self.sub_iterators,
                     self.directions))

    @classmethod
    def merge(cls, *others):
        if not others:
            return IterationSpace(IntervalGroup())
        intervals = IntervalGroup.generate('merge', *[i.intervals for i in others])
        directions = {}
        for i in others:
            for k, v in i.directions.items():
                if directions.get(k, Any) in (Any, v):
                    # No direction yet, or Any, or simply identical to /v/
                    directions[k] = v
                elif v is not Any:
                    # Clash detected
                    raise ValueError("Cannot merge `IterationSpace`s with "
                                     "incompatible directions")
        sub_iterators = {}
        for i in others:
            for k, v in i.sub_iterators.items():
                cv = [j.copy() for j in v]
                ret = sub_iterators.setdefault(k, cv)
                for se in cv:
                    ofs = dict(ret).get(se.dim)
                    if ofs is None:
                        ret.append(se)
                    else:
                        ofs.update(se.ofs)
        return IterationSpace(intervals, sub_iterators, directions)

    def project(self, cond):
        """Return a new ``IterationSpace`` in which only some :class:`Dimension`s
        in ``self`` are retained. In particular, a dimension ``d`` in ``self`` is
        retained if: ::

            * either ``cond(d)`` is true (``cond`` is a callable),
            * or ``d in cond`` is true (``cond`` is an iterable)
        """
        if callable(cond):
            func = cond
        else:
            func = lambda i: i in cond
        intervals = [i for i in self.intervals if func(i.dim)]
        sub_iterators = {k: v for k, v in self.sub_iterators.items() if func(k)}
        directions = {k: v for k, v in self.directions.items() if func(k)}
        return IterationSpace(intervals, sub_iterators, directions)

    def is_compatible(self, other):
        """A relaxed version of ``__eq__``, in which only non-derived dimensions
        are compared for equality."""
        return self.intervals == other.intervals and\
            self.nonderived_directions == other.nonderived_directions

    @property
    def sub_iterators(self):
        return self._sub_iterators

    @property
    def directions(self):
        return self._directions

    @property
    def iteration_intervals(self):
        return tuple(IterationInterval(i, self.directions[i.dim]) for i in self.intervals)

    @property
    def args(self):
        return (self.intervals, self.sub_iterators, self.directions)

    @property
    def dimensions(self):
        sub_dims = [i.dim for v in self.sub_iterators.values() for i in v]
        return filter_ordered(self.intervals.dimensions + sub_dims)

    @property
    def nonderived_directions(self):
        return {k: v for k, v in self.directions.items() if not k.is_Derived}

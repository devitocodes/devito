import abc
from collections import OrderedDict
from functools import reduce, cached_property
from operator import mul

from sympy import Expr

from devito.ir.support.utils import minimum, maximum
from devito.ir.support.vector import Vector, vmin, vmax
from devito.tools import (Ordering, Stamp, as_list, as_set, as_tuple,
                          filter_ordered, flatten, frozendict, is_integer,
                          toposort)
from devito.types import Dimension, ModuloDimension

__all__ = ['NullInterval', 'Interval', 'IntervalGroup', 'IterationSpace',
           'IterationInterval', 'DataSpace', 'Forward', 'Backward', 'Any',
           'null_ispace']


# The default Stamp, used by all new Intervals
S0 = Stamp()


class AbstractInterval:

    """
    An abstract representation of an iterated closed interval on Z.
    """

    __metaclass__ = abc.ABCMeta

    is_Null = False
    is_Defined = False

    def __init__(self, dim, stamp=S0):
        self.dim = dim
        self.stamp = stamp

    def __eq__(self, o):
        return (type(self) is type(o) and
                self.dim is o.dim and
                self.stamp == o.stamp)

    is_compatible = __eq__

    def __hash__(self):
        return hash(self.dim.name)

    @abc.abstractmethod
    def _rebuild(self):
        return

    @property
    @abc.abstractmethod
    def relaxed(self):
        return

    def intersection(self, o):
        return self._rebuild()

    @abc.abstractmethod
    def union(self, o):
        return self._rebuild()

    def add(self, o):
        return self._rebuild()

    subtract = add

    def negate(self):
        return self._rebuild()

    zero = negate
    flip = negate
    lift = negate
    reset = negate
    switch = negate
    translate = negate


class NullInterval(AbstractInterval):

    """
    A degenerate iterated closed interval on Z.
    """

    is_Null = True

    def __repr__(self):
        return "%s[Null]%s" % (self.dim, self.stamp)

    def __hash__(self):
        return hash(self.dim)

    def _rebuild(self):
        return NullInterval(self.dim, self.stamp)

    @property
    def relaxed(self):
        return NullInterval(self.dim.root, self.stamp)

    def union(self, o):
        if self.dim is o.dim:
            return o._rebuild()
        else:
            raise ValueError("Cannot compute union of Intervals over "
                             "different Dimensions")

    def switch(self, d):
        return NullInterval(d, self.stamp)


class Interval(AbstractInterval):

    """
    Interval(dim, lower, upper)

    A concrete iterated closed interval on Z.

    An Interval defines the compact region

        [dim.symbolic_min + lower, dim.symbolic_max + upper]
    """

    is_Defined = True

    def __init__(self, dim, lower=0, upper=0, stamp=S0):
        super().__init__(dim, stamp)

        try:
            self.lower = int(lower)
        except TypeError:
            assert isinstance(lower, Expr)
            self.lower = lower
        try:
            self.upper = int(upper)
        except TypeError:
            assert isinstance(upper, Expr)
            self.upper = upper

    def __repr__(self):
        return "%s[%s,%s]%s" % (self.dim, self.lower, self.upper, self.stamp)

    def __hash__(self):
        return hash((self.dim, self.offsets))

    def __eq__(self, o):
        if self is o:
            return True

        return (super().__eq__(o) and
                self.lower == o.lower and
                self.upper == o.upper)

    def _rebuild(self):
        return Interval(self.dim, self.lower, self.upper, self.stamp)

    @property
    def symbolic_min(self):
        return self.dim.symbolic_min + self.lower

    @property
    def symbolic_max(self):
        return self.dim.symbolic_max + self.upper

    @cached_property
    def size(self):
        """
        The Interval size, defined as the number of points iterated over through
        ``self.dim``, namely

            (dim.symbolic_max + upper - dim.symbolic_min - lower + 1) / dim.symbolic_incr

        Notes
        -----
        The Interval size is typically a function of several symbols (e.g.,
        `self.dim.symbolic_max`), and all such symbols must be mappable
        to actual numbers at `op.apply` time (i.e., the runtime values).
        When `self.dim` is an "unstructured Dimension", such as ModuloDimension,
        things can get nasty since the symbolic min/max/incr can literally be
        anything (any expression involving any Dimension/symbol/...), which makes
        it extremely complicated to numerically compute the size. However,
        the compiler only uses such unstructured Dimensions in well defined
        circumstances, which we explicitly handle here. Ultimately, therefore,
        we return a `size` that is made up of known symbols.
        """
        if self.dim.is_Custom and isinstance(self.dim.symbolic_min, ModuloDimension):
            # Special case 1)
            # Caused by the performance option `cire-rotate=True`
            d = self.dim.symbolic_min
            n = d.parent.symbolic_size

            # Iteration 0:
            assert is_integer(d.symbolic_min)
            assert is_integer(d.symbolic_incr)
            assert is_integer(self.dim.symbolic_max)
            assert self.lower == self.upper
            npoints = self.dim.symbolic_max - d.symbolic_min + 1
            # Iterations [1, ..., n-1]:
            assert d.symbolic_incr == self.dim.symbolic_max
            npoints += 1 * (n-1)
            npoints /= n
        else:
            # Typically we end up here (Dimension, SubDimension, BlockDimension)
            upper_extreme = self.symbolic_max
            lower_extreme = self.symbolic_min

            npoints = (upper_extreme - lower_extreme + 1)

        return npoints / self.dim.symbolic_incr

    @cached_property
    def min_size(self):
        return self.upper - self.lower + 1

    @property
    def relaxed(self):
        return Interval(self.dim.root, self.lower, self.upper, self.stamp)

    @property
    def offsets(self):
        return (self.lower, self.upper)

    def intersection(self, o):
        if self.is_compatible(o):
            svl, svu = Vector(self.lower, smart=True), Vector(self.upper, smart=True)
            ovl, ovu = Vector(o.lower, smart=True), Vector(o.upper, smart=True)
            return Interval(self.dim, vmax(svl, ovl)[0], vmin(svu, ovu)[0], self.stamp)
        else:
            return NullInterval(self.dim)

    def union(self, o):
        if o.is_Null and self.dim is o.dim:
            return self._rebuild()
        elif self.is_compatible(o):
            svl, svu = Vector(self.lower, smart=True), Vector(self.upper, smart=True)
            ovl, ovu = Vector(o.lower, smart=True), Vector(o.upper, smart=True)
            return Interval(self.dim, vmin(svl, ovl)[0], vmax(svu, ovu)[0], self.stamp)
        else:
            raise ValueError("Cannot compute union of non-compatible Intervals (%s, %s)" %
                             (self, o))

    def add(self, o):
        if not self.is_compatible(o):
            return self._rebuild()
        else:
            return Interval(self.dim, self.lower + o.lower, self.upper + o.upper,
                            self.stamp)

    def subtract(self, o):
        if not self.is_compatible(o):
            return self._rebuild()
        else:
            return Interval(self.dim, self.lower - o.lower, self.upper - o.upper,
                            self.stamp)

    def negate(self):
        return Interval(self.dim, -self.lower, -self.upper, self.stamp)

    def zero(self):
        return Interval(self.dim, 0, 0, self.stamp)

    def flip(self):
        return Interval(self.dim, self.upper, self.lower, self.stamp)

    def lift(self, v=None):
        if v is None:
            v = Stamp()
        return Interval(self.dim, self.lower, self.upper, v)

    def reset(self):
        return Interval(self.dim, self.lower, self.upper, S0)

    def switch(self, d):
        return Interval(d, self.lower, self.upper, self.stamp)

    def translate(self, v0=0, v1=None):
        if v1 is None:
            v1 = v0
        return Interval(self.dim, self.lower + v0, self.upper + v1, self.stamp)

    def promote(self, cond):
        if cond(self.dim):
            try:
                return self.switch(self.dim.parent).promote(cond)
            except AttributeError:
                pass
        return self

    def expand(self):
        return Interval(
            self.dim, minimum(self.lower), maximum(self.upper), self.stamp
        )


class IntervalGroup(Ordering):

    """
    A sequence of Intervals equipped with set-like operations.
    """

    @classmethod
    def reorder(cls, items, relations):
        if not all(isinstance(i, AbstractInterval) for i in items):
            raise ValueError("Cannot create IntervalGroup from objs of type [%s]" %
                             ', '.join(str(type(i)) for i in items))

        if len(relations) == 1:
            # Special case: avoid expensive topological sorting if possible
            relation, = relations
            ordering = filter_ordered(list(relation) + [i.dim for i in items])
        else:
            ordering = filter_ordered(toposort(relations) + [i.dim for i in items])

        return sorted(items, key=lambda i: ordering.index(i.dim))

    @classmethod
    def simplify_relations(cls, relations, items, mode):
        if mode == 'total':
            return [tuple(i.dim for i in items)]
        else:
            return super().simplify_relations(relations, items, mode)

    def __eq__(self, o):
        return len(self) == len(o) and all(i == j for i, j in zip(self, o))

    def __contains__(self, d):
        return any(i.dim is d for i in self)

    def __hash__(self):
        return hash(tuple(self))

    def __repr__(self):
        return "IntervalGroup[%s]" % (', '.join([repr(i) for i in self]))

    @cached_property
    def dimensions(self):
        return tuple(filter_ordered([i.dim for i in self]))

    @property
    def size(self):
        if self:
            return reduce(mul, [i.size for i in self])
        else:
            return 0

    @cached_property
    def is_well_defined(self):
        """
        True if all Intervals are over different Dimensions,
        False otherwise.
        """
        return len(self.dimensions) == len(set(self.dimensions))

    @classmethod
    def generate(cls, op, *interval_groups, relations=None):
        """
        Create a new IntervalGroup from the iterative application of an
        operation to some IntervalGroups.

        Parameters
        ----------
        op : str
            Any legal Interval operation, such as 'intersection' or
            or 'union'.
        *interval_groups
            Input IntervalGroups.
        relations : tuple, optional
            Additional relations to build the new IntervalGroup.

        Examples
        --------
        >>> from devito import dimensions
        >>> x, y, z = dimensions('x y z')
        >>> ig0 = IntervalGroup([Interval(x, 1, -1)])
        >>> ig1 = IntervalGroup([Interval(x, 2, -2), Interval(y, 3, -3)])
        >>> ig2 = IntervalGroup([Interval(y, 2, -2), Interval(z, 1, -1)])

        >>> ig = IntervalGroup.generate('intersection', ig0, ig1, ig2)
        """
        if op == 'intersection':
            dims = set.intersection(*[set(ig.dimensions) for ig in interval_groups])
        else:
            dims = set().union(*[ig.dimensions for ig in interval_groups])

        mapper = {}
        for ig in interval_groups:
            for i in ig:
                if i.dim in dims:
                    mapper.setdefault(i.dim, []).append(i)

        intervals = []
        for v in mapper.values():
            # Create a new Interval through the concatenation v0.key(v1).key(v2)...
            interval = v[0]
            for i in v[1:]:
                interval = getattr(interval, op)(i)
            intervals.append(interval)

        relations = as_set(relations)
        relations.update(set().union(*[ig.relations for ig in interval_groups]))

        modes = set(ig.mode for ig in interval_groups)
        assert len(modes) <= 1
        try:
            mode = modes.pop()
        except KeyError:
            mode = 'total'

        return IntervalGroup(intervals, relations=relations, mode=mode)

    def is_compatible(self, o):
        """
        Two IntervalGroups are compatible iff their elements are not subjected
        to sets of relations that would prevent a reordering.
        """
        if set(self) != set(o):
            return False
        if all(i == j for i, j in zip(self, o)):
            # Same input ordering, definitely compatible
            return True
        try:
            self.add(o)
            return True
        except ValueError:
            # Cyclic dependence detected
            return False

    def _normalize(func):
        """
        A simple decorator to normalize the input of operator methods that
        expect an IntervalGroup as an operand.
        """
        def wrapper(self, o):
            if not isinstance(o, IntervalGroup):
                o = IntervalGroup(as_tuple(o))
            if self.mode != o.mode:
                raise ValueError("Incompatible ordering modes")
            relations = self.relations | o.relations
            return func(self, o, relations, self.mode)
        return wrapper

    @_normalize
    def intersection(self, o, relations, mode):
        mapper = OrderedDict([(i.dim, i) for i in o])
        intervals = [i.intersection(mapper.get(i.dim, i)) for i in self]
        return IntervalGroup(intervals, relations=relations, mode=mode)

    @_normalize
    def add(self, o, relations, mode):
        mapper = OrderedDict([(i.dim, i) for i in o])
        intervals = [i.add(mapper.get(i.dim, NullInterval(i.dim))) for i in self]
        return IntervalGroup(intervals, relations=relations, mode=mode)

    @_normalize
    def subtract(self, o, relations, mode):
        mapper = OrderedDict([(i.dim, i) for i in o])
        intervals = [i.subtract(mapper.get(i.dim, NullInterval(i.dim))) for i in self]
        return IntervalGroup(intervals, relations=relations, mode=mode)

    def relaxed(self, d=None):
        d = set(self.dimensions if d is None else as_tuple(d))
        intervals = [i.relaxed if i.dim in d else i for i in self]
        return IntervalGroup(intervals, relations=self.relations, mode=self.mode)

    def promote(self, cond, mode='unordered'):
        intervals = [i.promote(cond) for i in self]
        intervals = IntervalGroup(intervals, relations=None, mode=mode)

        # There could be duplicate Dimensions at this point, so we sum up the
        # Intervals defined over the same Dimension to produce a well-defined
        # IntervalGroup
        intervals = IntervalGroup.generate('add', intervals)

        return intervals

    def drop(self, maybe_callable, strict=False):
        if callable(maybe_callable):
            dims = {i.dim for i in self if maybe_callable(i.dim)}
        else:
            dims = as_set(maybe_callable)

        # Dropping
        if strict:
            intervals = [i._rebuild() for i in self if i.dim not in dims]
        else:
            dims = set().union(*[i._defines for i in dims])
            intervals = [i._rebuild() for i in self if not i.dim._defines & dims]

        # Clean up relations
        relations = [tuple(i for i in r if i not in dims) for r in self.relations]

        return IntervalGroup(intervals, relations=relations, mode=self.mode)

    def negate(self):
        intervals = [i.negate() for i in self]

        return IntervalGroup(intervals, relations=self.relations, mode=self.mode)

    def zero(self, d=None):
        d = self.dimensions if d is None else as_tuple(d)
        intervals = [i.zero() if i.dim in d else i for i in self]

        return IntervalGroup(intervals, relations=self.relations, mode=self.mode)

    def lift(self, d=None, v=None):
        d = set(self.dimensions if d is None else as_tuple(d))
        intervals = [i.lift(v) if i.dim._defines & d else i for i in self]

        return IntervalGroup(intervals, relations=self.relations, mode=self.mode)

    def reset(self):
        intervals = [i.reset() for i in self]

        return IntervalGroup(intervals, relations=self.relations, mode=self.mode)

    def switch(self, d0, d1):
        intervals = [i.switch(d1) if i.dim is d0 else i for i in self]

        # Update relations too
        relations = {tuple(d1 if i is d0 else i for i in r) for r in self.relations}

        return IntervalGroup(intervals, relations=relations, mode=self.mode)

    def translate(self, d, v0=0, v1=None):
        dims = as_tuple(d)
        intervals = [i.translate(v0, v1) if i.dim in dims else i for i in self]

        return IntervalGroup(intervals, relations=self.relations, mode=self.mode)

    def expand(self, d=None):
        if d is None:
            d = self.dimensions
        intervals = [i.expand() if i.dim in as_tuple(d) else i for i in self]

        return IntervalGroup(intervals, relations=self.relations, mode=self.mode)

    def index(self, key):
        if isinstance(key, Interval):
            return super().index(key)
        elif isinstance(key, Dimension):
            return super().index(self[key])
        raise ValueError("Expected Interval or Dimension, got `%s`" % type(key))

    def __getitem__(self, key):
        if is_integer(key):
            return super().__getitem__(key)
        elif isinstance(key, slice):
            retval = super().__getitem__(key)
            return IntervalGroup(retval, relations=self.relations, mode=self.mode)

        if not self.is_well_defined:
            raise ValueError("Cannot fetch Interval from ill defined Space")

        if not isinstance(key, Dimension):
            return NullInterval(key)

        for i in self:
            # Obvious case
            if i.dim is key:
                return i

            # There are a few special cases for which we can support
            # derived-Dimension-based indexing
            if key.is_NonlinearDerived and i.dim in key._defines:
                return i
            if i.dim.is_Custom and key in i.dim._defines:
                return i

        return NullInterval(key)


class IterationDirection:

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


class IterationInterval(Interval):

    """
    An Interval associated with metadata.
    """

    def __init__(self, interval, sub_iterators=(), direction=Forward):
        super().__init__(interval.dim, *interval.offsets, stamp=interval.stamp)
        self.sub_iterators = sub_iterators
        self.direction = direction

    def __repr__(self):
        return "%s%s" % (super().__repr__(), self.direction)

    def __eq__(self, other):
        if not isinstance(other, IterationInterval):
            return False
        return self.direction is other.direction and super().__eq__(other)

    def __hash__(self):
        return hash((self.dim, self.offsets, self.direction))

    @property
    def args(self):
        return (self, self.sub_iterators, self.direction)


class Space:

    """
    A compact N-dimensional space defined by N Intervals.

    Parameters
    ----------
    intervals : tuple of Intervals
        Space description.
    """

    def __init__(self, intervals):
        if isinstance(intervals, IntervalGroup):
            self._intervals = intervals
        else:
            self._intervals = IntervalGroup(as_tuple(intervals))

    def __repr__(self):
        return "%s[%s]" % (self.__class__.__name__,
                           ", ".join(repr(i) for i in self.intervals))

    def __eq__(self, other):
        return isinstance(other, Space) and self.intervals == other.intervals

    def __hash__(self):
        return hash(self.intervals)

    def __len__(self):
        return len(self.intervals)

    def __iter__(self):
        for i in self.intervals:
            yield i

    @property
    def intervals(self):
        return self._intervals

    @cached_property
    def dimensions(self):
        return tuple(filter_ordered(self.intervals.dimensions))

    @property
    def size(self):
        return self.intervals.size

    @property
    def mode(self):
        return self.intervals.mode

    @property
    def relations(self):
        return self.intervals.relations

    @property
    def dimension_map(self):
        """
        Map between the Space Dimensions and the size of their Interval.
        """
        return OrderedDict([(i.dim, i.size) for i in self.intervals])


class DataSpace(Space):

    """
    Represent a data space as a Space with additional metadata and operations.

    Parameters
    ----------
    intervals : tuple of Intervals
        Data space description.
    parts : dict, optional
        A mapper from Functions to IntervalGroup, describing the individual
        components of the data space.
    """

    def __init__(self, intervals, parts=None):
        super().__init__(intervals)

        # Normalize parts
        parts = {k: v.expand() for k, v in (parts or {}).items()}
        for k, v in list(parts.items()):
            dims = set().union(*[d._defines for d in k.dimensions])
            parts[k] = v.drop(lambda d: d not in dims)
        self._parts = frozendict(parts)

    def __eq__(self, other):
        return (isinstance(other, DataSpace) and
                self.intervals == other.intervals and
                self.parts == other.parts)

    def __hash__(self):
        return hash((super().__hash__(), self.parts))

    @classmethod
    def union(cls, *others):
        if not others:
            return DataSpace(IntervalGroup(), {})

        intervals = IntervalGroup.generate('union', *[i.intervals for i in others])

        parts = {}
        for i in others:
            for k, v in i.parts.items():
                parts.setdefault(k, []).append(v)
        parts = {k: IntervalGroup.generate('union', *v) for k, v in parts.items()}

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

    def reset(self):
        intervals = self.intervals.reset()
        parts = {k: v.reset() for k, v in self.parts.items()}

        return DataSpace(intervals, parts)


class IterationSpace(Space):

    """
    Represent an iteration space as a Space with additional metadata and operations.

    Parameters
    ----------
    intervals : IntervalGroup
        Iteration space description.
    sub_iterators : dict, optional
        A mapper from Dimensions in ``intervals`` to iterables of
        DerivedDimensions defining sub-regions of iteration.
    directions : dict, optional
        A mapper from Dimensions in ``intervals`` to IterationDirections.
    """

    def __init__(self, intervals, sub_iterators=None, directions=None):
        super().__init__(intervals)

        # Normalize sub-iterators
        sub_iterators = dict([(k, tuple(filter_ordered(as_tuple(v))))
                              for k, v in (sub_iterators or {}).items()])
        sub_iterators.update({i.dim: () for i in self.intervals
                              if i.dim not in sub_iterators})
        self._sub_iterators = frozendict(sub_iterators)

        # Normalize directions
        if directions is None:
            self._directions = frozendict([(i.dim, Any) for i in self.intervals])
        else:
            self._directions = frozendict(directions)

    def __repr__(self):
        ret = ', '.join(["%s%s" % (repr(i), repr(self.directions[i.dim]))
                         for i in self.intervals])
        return "IterationSpace[%s]" % ret

    def __eq__(self, other):
        if self is other:
            return True

        return (isinstance(other, IterationSpace) and
                self.intervals == other.intervals and
                self.directions == other.directions)

    def __lt__(self, other):
        """
        A rudimentary less-then comparison between two IterationSpaces.
        """
        return len(self.itintervals) < len(other.itintervals)

    def __hash__(self):
        return hash((super().__hash__(), self.sub_iterators,
                     self.directions))

    def __contains__(self, d):
        try:
            self[d]
            return True
        except KeyError:
            return False

    def __getitem__(self, key):
        v = self.intervals[key]
        if isinstance(key, slice):
            return self.project(lambda d: d in v.dimensions)
        else:
            d = v.dim
            return IterationInterval(v, self.sub_iterators[d], self.directions[d])

    @classmethod
    def generate(self, op, *others, relations=None):
        if not others:
            return IterationSpace(IntervalGroup())
        elif len(others) == 1 and not relations:
            return others[0]

        intervals = [i.intervals for i in others]
        intervals = IntervalGroup.generate(op, *intervals, relations=relations)

        directions = {}
        for i in others:
            for k, v in i.directions.items():
                if directions.get(k, Any) in (Any, v):
                    # No direction yet, or Any, or simply identical to /v/
                    directions[k] = v
                elif v is not Any:
                    # Clash detected
                    raise ValueError("Cannot compute %s of `IterationSpace`s "
                                     "with incompatible directions" % op)

        sub_iterators = {}
        for i in others:
            for k, v in i.sub_iterators.items():
                ret = sub_iterators.setdefault(k, [])
                ret.extend([d for d in v if d not in ret])

        return IterationSpace(intervals, sub_iterators, directions)

    @classmethod
    def union(cls, *others, relations=None):
        return cls.generate('union', *others, relations=relations)

    @classmethod
    def intersection(cls, *others, relations=None):
        return cls.generate('intersection', *others, relations=relations)

    def index(self, key):
        return self.intervals.index(key)

    def add(self, other):
        return IterationSpace(self.intervals.add(other), self.sub_iterators,
                              self.directions)

    def augment(self, sub_iterators):
        """
        Create a new IterationSpace with additional sub-iterators.
        """
        items = dict(self.sub_iterators)
        for k, v in sub_iterators.items():
            if k not in self.intervals:
                continue
            items[k] = as_list(items.get(k))
            for i in as_list(v):
                if i not in items[k]:
                    items[k].append(i)

        return IterationSpace(self.intervals, items, self.directions)

    def switch(self, d0, d1, direction=None):
        intervals = self.intervals.switch(d0, d1)

        sub_iterators = dict(self.sub_iterators)
        sub_iterators.pop(d0, None)
        sub_iterators[d1] = ()

        directions = dict(self.directions)
        v = directions.pop(d0, None)
        directions[d1] = direction or v

        return IterationSpace(intervals, sub_iterators, directions)

    def translate(self, d, v0=0, v1=None):
        intervals = self.intervals.translate(d, v0, v1)
        return IterationSpace(intervals, self.sub_iterators, self.directions)

    def reset(self):
        return IterationSpace(self.intervals.reset(), self.sub_iterators, self.directions)

    def project(self, cond, strict=True):
        """
        Create a new IterationSpace retaining only some of the Dimensions in
        `self`. In particular, a Dimension `d` in `self` is retained if:

            * either `cond(d)` is true (`cond` is a callable),
            * or `d in cond` is true (`cond` is an iterable)
        """
        if callable(cond):
            func = cond
        else:
            func = lambda i: i in cond

        dims = [i.dim for i in self if not func(i.dim)]
        intervals = self.intervals.drop(dims, strict=strict)

        sub_iterators = {}
        directions = {}
        for i in intervals:
            d = i.dim
            sub_iterators[d] = self.sub_iterators[d]
            directions[d] = self.directions[d]

        return IterationSpace(intervals, sub_iterators, directions)

    def zero(self, d=None):
        intervals = self.intervals.zero(d)

        return IterationSpace(intervals, self.sub_iterators, self.directions)

    def lift(self, d=None, v=None):
        intervals = self.intervals.lift(d, v)

        return IterationSpace(intervals, self.sub_iterators, self.directions)

    def relaxed(self, cond):
        """
        Create a new IterationSpace in which certain DerivedDimensions
        are replaced with their roots. In particular, a Dimension `d` in
        `self` is relaxed, and therefore replaced with `d.root`, if:

            * either `cond(d)` is true (`cond` is a callable),
            * or `d in cond` is true (`cond` is an iterable)
        """
        if callable(cond):
            dims = tuple(i.dim for i in self if cond(i.dim))
        else:
            dims = as_tuple(cond)

        intervals = self.intervals.relaxed(dims)
        sub_iterators = {d.root if d in dims else d: v
                         for d, v in self.sub_iterators.items()}
        directions = {d.root if d in dims else d: v
                      for d, v in self.directions.items()}

        return IterationSpace(intervals, sub_iterators, directions)

    def promote(self, cond, mode='unordered'):
        intervals = self.intervals.promote(cond, mode=mode)
        sub_iterators = {i.promote(cond).dim: self.sub_iterators[i.dim]
                         for i in self.intervals}
        directions = {i.promote(cond).dim: self.directions[i.dim]
                      for i in self.intervals}

        return IterationSpace(intervals, sub_iterators, directions)

    def prefix(self, key):
        """
        Return the IterationSpace up to and including the last Interval
        such that `key(interval)` is True.
        """
        try:
            i = self.project(key)[-1]
        except IndexError:
            return null_ispace

        return self[:self.index(i.dim) + 1]

    def split(self, d):
        """
        Split the IterationSpace into two IterationSpaces, the first
        containing all Intervals up to and including the one defined
        over Dimension `d`, and the second containing all Intervals starting
        from the one defined over Dimension `d`.
        """
        if isinstance(d, IterationInterval):
            d = d.dim

        if d is None:
            return IterationSpace([]), self

        try:
            n = self.index(d) + 1
            return self[:n], self[n:]
        except ValueError:
            return self, IterationSpace([])

    def insert(self, d, dimensions, sub_iterators=None, directions=None):
        """
        Insert new Dimensions into the IterationSpace after Dimension `d`.
        """
        dimensions = as_tuple(dimensions)

        intervals = IntervalGroup([Interval(i) for i in dimensions])
        ispace = IterationSpace(intervals, sub_iterators, directions)

        ispace0, ispace1 = self.split(d)
        relations = (ispace0.itdims + dimensions, dimensions + ispace1.itdims)

        return IterationSpace.union(ispace0, ispace, ispace1, relations=relations)

    def reorder(self, relations=None, mode=None):
        relations = relations or self.relations
        mode = mode or self.mode
        intervals = IntervalGroup(self.intervals, relations=relations, mode=mode)

        return IterationSpace(intervals, self.sub_iterators, self.directions)

    def is_subset(self, other):
        """
        True if `self` is included within `other`, False otherwise.
        """
        if not self:
            return True

        d = self[-1].dim
        try:
            return self == other[:other.index(d) + 1]
        except ValueError:
            return False

    def is_compatible(self, other):
        """
        A relaxed version of ``__eq__``, in which only non-derived dimensions
        are compared for equality.
        """
        return (self.intervals.is_compatible(other.intervals) and
                self.nonderived_directions == other.nonderived_directions)

    @property
    def itdims(self):
        return self.intervals.dimensions

    @property
    def sub_iterators(self):
        return self._sub_iterators

    @property
    def directions(self):
        return self._directions

    @property
    def outermost(self):
        return self[0]

    @property
    def innermost(self):
        return self[-1]

    @cached_property
    def concrete(self):
        return self.project(lambda d: not d.is_Virtual)

    @cached_property
    def itintervals(self):
        return tuple(self[d] for d in self.itdims)

    @cached_property
    def dimensions(self):
        sub_dims = flatten(i._defines for v in self.sub_iterators.values() for i in v)
        r_dims = flatten(i._defines for v in self.relations for i in v)
        return tuple(filter_ordered(self.itdims + tuple(sub_dims) + tuple(r_dims)))

    @cached_property
    def nonderived_directions(self):
        return {k: v for k, v in self.directions.items() if not k.is_Derived}


null_ispace = IterationSpace([])

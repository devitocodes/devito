from collections import OrderedDict

from cached_property import cached_property
from sympy import Basic, S

from devito.ir.support.space import Any, Backward
from devito.symbolics import retrieve_terminals, q_monoaffine
from devito.tools import (EnrichedTuple, Tag, as_tuple, is_integer,
                          filter_sorted, flatten, memoized_meth)
from devito.types import Dimension

__all__ = ['Vector', 'LabeledVector', 'IterationInstance', 'Access',
           'TimedAccess', 'Scope']


class Vector(tuple):

    """
    A representation of a vector in Z^n.

    The elements of a Vector can be integers or any SymPy expression.

    Notes
    -----
    1) Vector-scalar comparison
    If a comparison between a vector and a non-vector is attempted, then the
    non-vector is promoted to a vector; if this is not possible, an exception
    is raised. This is handy because it turns a vector-scalar comparison into
    a vector-vector comparison with the scalar broadcasted to all vector entries.
    For example: ::

        (3, 4, 5) > 4 => (3, 4, 5) > (4, 4, 4) => False

    2) Comparing Vector entries when these are SymPy expression
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

    def _asvector(relax=False):
        def __asvector(func):
            def wrapper(self, other):
                if not isinstance(other, Vector):
                    try:
                        other = Vector(*other)
                    except TypeError:
                        # Not iterable
                        other = Vector(*(as_tuple(other)*len(self)))
                if relax is False and len(self) != len(other):
                    raise TypeError("Cannot operate with Vectors of different rank")
                return func(self, other)
            return wrapper
        return __asvector

    @_asvector()
    def __add__(self, other):
        return Vector(*[i + j for i, j in zip(self, other)])

    @_asvector()
    def __radd__(self, other):
        return self + other

    @_asvector()
    def __sub__(self, other):
        return Vector(*[i - j for i, j in zip(self, other)])

    @_asvector()
    def __rsub__(self, other):
        return self - other

    @_asvector(relax=True)
    def __eq__(self, other):
        return super(Vector, self).__eq__(other)

    def __hash__(self):
        return super(Vector, self).__hash__()

    @_asvector(relax=True)
    def __ne__(self, other):
        return super(Vector, self).__ne__(other)

    @_asvector()
    def __lt__(self, other):
        # This might raise an exception if the distance between the i-th entry
        # of /self/ and /other/ isn't integer, but rather a generic function
        # (and thus not comparable to 0). However, the implementation is "smart",
        # in the sense that it will return as soon as the first two comparable
        # entries (i.e., such that their distance is a non-zero integer) are found
        for i in self.distance(other):
            try:
                val = int(i)
            except TypeError:
                raise TypeError("Cannot compare due to non-comparable index functions")
            if val < 0:
                return True
            elif val > 0:
                return False

    @_asvector()
    def __gt__(self, other):
        return other.__lt__(self)

    @_asvector()
    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    @_asvector()
    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __getitem__(self, key):
        ret = super(Vector, self).__getitem__(key)
        return Vector(*ret) if isinstance(key, slice) else ret

    def __repr__(self):
        return "(%s)" % ','.join(str(i) for i in self)

    @property
    def rank(self):
        return len(self)

    @property
    def sum(self):
        return sum(self)

    @property
    def is_constant(self):
        return all(is_integer(i) for i in self)

    def distance(self, other):
        """
        Compute the distance from ``self`` to ``other``.

        The distance is a reflexive, transitive, and anti-symmetric relation,
        which establishes a total ordering amongst Vectors.

        The distance is a function [Vector x Vector --> D]. D is a tuple of length
        equal to the Vector ``rank``. The i-th entry of D, D_i, indicates whether
        the i-th component of ``self``, self_i, precedes (< 0), equals (== 0), or
        succeeds (> 0) the i-th component of ``other``, other_i.

        In particular, the *absolute value* of D_i represents the number of
        integer points that exist between self_i and sink_i.

        Examples
        --------
                 | 3 |           | 1 |               |  2  |
        source = | 2 | ,  sink = | 4 | , distance => | -2  |
                 | 1 |           | 5 |               | -4  |

        There are 2, 2, and 4 points between [3-2], [2-4], and [1-5], respectively.
        """
        return self - other


class LabeledVector(Vector):

    """
    A Vector that associates a Dimension to each element.
    """

    def __new__(cls, items=None):
        try:
            labels, values = zip(*items)
        except (ValueError, TypeError):
            labels, values = (), ()
        if not all(isinstance(i, Dimension) for i in labels):
            raise ValueError("All labels must be of type Dimension, got [%s]"
                             % ','.join(i.__class__.__name__ for i in labels))
        obj = super(LabeledVector, cls).__new__(cls, *values)
        obj.labels = labels
        return obj

    @classmethod
    def transpose(cls, *vectors):
        """
        Transpose a matrix represented as an iterable of homogeneous LabeledVectors.
        """
        if len(vectors) == 0:
            return LabeledVector()
        if not all(isinstance(v, LabeledVector) for v in vectors):
            raise ValueError("All items must be of type LabeledVector, got [%s]"
                             % ','.join(i.__class__.__name__ for i in vectors))
        T = OrderedDict()
        for v in vectors:
            for l, i in zip(v.labels, v):
                T.setdefault(l, []).append(i)
        return tuple((l, Vector(*i)) for l, i in T.items())

    def __repr__(self):
        return "(%s)" % ','.join('%s:%s' % (l, i) for l, i in zip(self.labels, self))

    def __hash__(self):
        return hash((tuple(self), self.labels))

    def __eq__(self, other):
        if isinstance(other, LabeledVector) and self.labels != other.labels:
            raise TypeError("Cannot compare due to mismatching `labels`")
        return super(LabeledVector, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, LabeledVector) and self.labels != other.labels:
            raise TypeError("Cannot compare due to mismatching `labels`")
        return super(LabeledVector, self).__lt__(other)

    def __gt__(self, other):
        return other.__lt__(self)

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __getitem__(self, index):
        if isinstance(index, (slice, int)):
            return super(LabeledVector, self).__getitem__(index)
        elif isinstance(index, Dimension):
            for d in index._defines:
                if d in self.labels:
                    i = self.labels.index(d)
                    return super(LabeledVector, self).__getitem__(i)
            return None
        else:
            raise TypeError("Indices must be integers, slices, or Dimensions, not %s"
                            % type(index))

    def fromlabel(self, label, v=None):
        return self[label] if label in self.labels else v

    @memoized_meth
    def distance(self, other):
        """
        Compute the distance from ``self`` to ``other``.

        Parameters
        ----------
        other : LabeledVector
            The LabeledVector from which the distance is computed.
        """
        if not isinstance(other, LabeledVector):
            raise TypeError("Cannot compute distance from obj of type %s", type(other))
        if self.labels != other.labels:
            raise TypeError("Cannot compute distance due to mismatching `labels`")
        return LabeledVector(list(zip(self.labels, self - other)))


class IndexMode(Tag):
    """Tag for access functions."""
    pass
AFFINE = IndexMode('affine')  # noqa
IRREGULAR = IndexMode('irregular')


class IterationInstance(LabeledVector):

    """
    A representation of the iteration and data points accessed by an
    Indexed object. Three different concepts are distinguished:

        * Index functions: the expressions describing what *iteration* space point
          are accessed.
        * ``aindices``: the Dimensions acting as iteration variables.
          There is one aindex for each index function. If the index function
          is non-affine, then it may not be possible to detect its aindex;
          in such a case, None is used as placeholder.
        * ``findices``: the Dimensions describing what *data* space point
          are accessed.
    """

    def __new__(cls, indexed):
        findices = tuple(indexed.function.indices)
        if len(findices) != len(set(findices)):
            raise ValueError("Illegal non-unique `findices`")
        return super(IterationInstance, cls).__new__(cls,
                                                     list(zip(findices, indexed.indices)))

    def __hash__(self):
        return super(IterationInstance, self).__hash__()

    @cached_property
    def _cached_findices_index(self):
        # Avoiding to call self.findices.index repeatedly speeds analysis up
        return {fi: i for i, fi in enumerate(self.findices)}

    @cached_property
    def index_mode(self):
        index_mode = []
        for i, fi in zip(self, self.findices):
            if q_monoaffine(i, fi, self.findices):
                index_mode.append(AFFINE)
            else:
                dims = {i for i in i.free_symbols if isinstance(i, Dimension)}
                try:
                    # There's still hope it's regular if a DerivedDimension is used
                    candidate = dims.pop()
                    if candidate.root == fi and q_monoaffine(i, candidate, self.findices):
                        index_mode.append(AFFINE)
                        continue
                except (KeyError, AttributeError):
                    pass
                index_mode.append(IRREGULAR)
        return tuple(index_mode)

    @cached_property
    def aindices(self):
        aindices = []
        for i, fi in zip(self, self.findices):
            if q_monoaffine(i, fi, self.findices):
                aindices.append(fi)
            else:
                dims = {i for i in i.free_symbols if isinstance(i, Dimension)}
                aindices.append(dims.pop() if len(dims) == 1 else None)
        return EnrichedTuple(*aindices, getters=self.findices)

    @property
    def findices(self):
        return self.labels

    @cached_property
    def findices_affine(self):
        return tuple(fi for fi, im in zip(self.findices, self.index_mode) if im == AFFINE)

    @cached_property
    def findices_irregular(self):
        return tuple(fi for fi, im in zip(self.findices, self.index_mode)
                     if im == IRREGULAR)

    def affine(self, findices):
        """
        Return True if all of the provided findices appear in self and are
        affine, False otherwise.
        """
        return set(as_tuple(findices)).issubset(set(self.findices_affine))

    def affine_if_present(self, findices):
        """
        Return False if any of the provided findices appears in self and
        is not affine, True otherwise.
        """
        findices = as_tuple(findices)
        return (set(findices) & set(self.findices)).issubset(set(self.findices_affine))

    def touched_halo(self, findex):
        """
        Return a boolean 2-tuple, one entry for each ``findex`` DataSide. True
        means that the halo is touched along that DataSide.
        """
        # If an irregularly (non-affine) accessed Dimension, conservatively
        # assume the halo will be touched
        if self.irregular(findex):
            return (True, True)

        aindex = self.aindices[findex]

        # If the iterator is *not* a distributed Dimension, then surely the
        # halo isn't touched
        try:
            if not aindex._maybe_distributed:
                return (False, False)
        except AttributeError:
            pass

        # Given `d` \in aindices, iterating over [0, size_d):
        # * if `self[d] - d < self.function._size_halo[d].left`, then `self` will
        #   definitely touch the left-halo when `d=0`
        # * if `self[d] - d > self.function._size_halo[d].left`, then `self` will
        #   definitely touch the right-halo when `d=size_d-1`
        size_halo_left = self.function._size_halo[findex].left
        try:
            touch_halo_left = bool(self[findex] - aindex < size_halo_left)
        except TypeError:
            # Conservatively assume True. We might end up here, for example,
            # in the following cases:
            # * The `aindex` doesn't appear in `self[findex]`, such as when the
            #   `aindex` is a pure number of a different Dimension
            # * `self[findex]` isn't affine in the `aindex`
            touch_halo_left = True
        try:
            touch_halo_right = bool(self[findex] - aindex > size_halo_left)
        except TypeError:
            # Same considerations as in the try-except above
            touch_halo_right = True
        return (touch_halo_left, touch_halo_right)

    def irregular(self, findices):
        """
        Return True if all of the provided findices appear in self and are
        irregular, False otherwise.
        """
        return set(as_tuple(findices)).issubset(set(self.findices_irregular))

    @property
    def is_regular(self):
        return all(i is AFFINE for i in self.index_mode)

    @property
    def is_irregular(self):
        return not self.is_regular

    @property
    def is_scalar(self):
        return self.rank == 0

    def distance(self, other, findex=None, view=None):
        """
        Compute the distance from ``self`` to ``other``.

        Parameters
        ----------
        other : IterationInstance
            The IterationInstance from which the distance is computed.
        findex : Dimension, optional
            If supplied, compute the distance only up to and including ``findex``.
        view : list of Dimension, optional
            If supplied, project the distance along these Dimensions.
        """
        if not isinstance(other, IterationInstance):
            raise TypeError("Cannot compute distance from obj of type %s", type(other))
        if self.findices != other.findices:
            raise TypeError("Cannot compute distance due to mismatching `findices`")
        if findex is not None:
            try:
                limit = self._cached_findices_index[findex] + 1
            except KeyError:
                raise TypeError("Cannot compute distance as `findex` not in `findices`")
        else:
            limit = self.rank
        distance = super(IterationInstance, self).distance(other)[:limit]
        if view is None:
            return distance
        else:
            proj = [d for d, i in zip(distance, self.findices) if i in as_tuple(view)]
            return Vector(*proj)

    def section(self, findices):
        """
        Return a view of ``self`` in which the slots corresponding to the
        provided ``findices`` have been zeroed.
        """
        return Vector(*[d if i not in as_tuple(findices) else 0
                        for d, i in zip(self, self.findices)])


class Access(IterationInstance):

    """
    A representation of the access performed by an Indexed object
    (a scalar in the degenerate case).

    Notes
    -----
    The comparison operators ``==, !=, <, <=, >, >=`` should be regarded as
    operators for lexicographic ordering of Access objects, based
    on the values of the index functions (and the index functions only).

    For example, if two Access objects A and B employ the same index functions,
    the operation A == B will return True regardless of whether A and B are
    reads or writes or mixed.
    """

    def __new__(cls, indexed, mode):
        assert mode in ['R', 'W', 'RI', 'WI']
        obj = super(Access, cls).__new__(cls, indexed)
        obj.indexed = indexed
        obj.function = indexed.function
        obj.mode = mode
        return obj

    def __eq__(self, other):
        return (isinstance(other, Access) and
                self.function == other.function and
                self.mode == other.mode and
                super(Access, self).__eq__(other))

    def __hash__(self):
        return super(Access, self).__hash__()

    @property
    def name(self):
        return self.function.name

    @property
    def is_read(self):
        return self.mode in ['R', 'RI']

    @property
    def is_write(self):
        return self.mode in ['W', 'WI']

    @property
    def is_read_increment(self):
        return self.mode == 'RI'

    @property
    def is_write_increment(self):
        return self.mode == 'WI'

    @property
    def is_increment(self):
        return self.is_read_increment or self.is_write_increment

    @property
    def is_local(self):
        return self.function.is_Symbol

    def __repr__(self):
        mode = '\033[1;37;31mW\033[0m' if self.is_write else '\033[1;37;32mR\033[0m'
        return "%s<%s,[%s]>" % (mode, self.name, ', '.join(str(i) for i in self))


class TimedAccess(Access):

    """
    A special Access object enriched with: ::

        * a "timestamp"; that is, an integer indicating the access location
          within the execution flow;
        * an array of directions; there is one direction for each index,
          indicating whether the index function is monotonically increasing
          or decreasing.

    Further, a TimedAccess may be regular or irregular. A TimedAccess is regular
    if and only if *all* index functions are affine in their respective findex.
    The downside of irregular TimedAccess objects is that dependence testing is
    harder, which in turn may force the data dependence analyzer to make stronger
    assumptions to be conservative.

    Examples
    --------
    Given:
    findices = [x, y, z]
    w = an object of type Dimension

           | x+1 |           |  x  |          |  x  |          | w |          | x+y |
    obj1 = | y+2 | ,  obj2 = |  4  | , obj3 = |  x  | , obj4 = | y | , obj5 = |  y  |
           | z-3 |           | z+1 |          |  y  |          | z |          |  z  |

    We have that: ::

        * obj1 and obj2 are regular;
        * obj3 is irregular because a findex, ``x``, appears outside of its index
          function (i.e., in the second slot, when ``y`` is expected);
        * obj4 is irregular, because a different dimension, ``w``, is used in place
          of ``x`` within the first index function, where ``x`` is expected;
        * obj5 is irregular, as two findices appear in the same index function --
          the one in the first slot, where only ``x`` is expected.
    """

    def __new__(cls, indexed, mode, timestamp, directions):
        assert is_integer(timestamp)
        obj = super(TimedAccess, cls).__new__(cls, indexed, mode)
        obj.timestamp = timestamp
        # We use `.root` as if a DerivedDimension is in `directions`, then so is
        # its parent, and the parent (root) direction cannot differ from that
        # of its child
        obj.directions = [directions.get(i.root, Any) for i in obj.findices]
        return obj

    def __eq__(self, other):
        return (isinstance(other, TimedAccess) and
                self.directions == other.directions and
                super(TimedAccess, self).__eq__(other))

    def __hash__(self):
        return super(TimedAccess, self).__hash__()

    def __lt__(self, other):
        if not isinstance(other, TimedAccess):
            raise TypeError("Cannot compare with object of type %s" % type(other))
        if self.directions != other.directions:
            raise TypeError("Cannot compare due to mismatching `direction`")
        return super(TimedAccess, self).__lt__(other)

    def lex_eq(self, other):
        return self.timestamp == other.timestamp

    def lex_ne(self, other):
        return self.timestamp != other.timestamp

    def lex_ge(self, other):
        return self.timestamp >= other.timestamp

    def lex_gt(self, other):
        return self.timestamp > other.timestamp

    def lex_le(self, other):
        return self.timestamp <= other.timestamp

    def lex_lt(self, other):
        return self.timestamp < other.timestamp

    def distance(self, other, findex=None):
        if self.rank != other.rank:
            raise TypeError("Cannot order due to mismatching `rank`")
        if not self.rank:
            return Vector()
        findex = findex or self.findices[-1]
        ret = []
        for i, sd, od in zip(self.findices, self.directions, other.directions):
            if sd == od:
                ret = list(super(TimedAccess, self).distance(other, i))
                if i == findex:
                    break
            else:
                ret.append(S.Infinity)
                break
        directions = self.directions[:self._cached_findices_index[i] + 1]
        assert len(directions) == len(ret)
        return Vector(*[(-i) if d == Backward else i for i, d in zip(ret, directions)])


class Dependence(object):

    """
    A data dependence between two TimedAccess objects.
    """

    def __init__(self, source, sink):
        assert isinstance(source, TimedAccess) and isinstance(sink, TimedAccess)
        assert source.function == sink.function
        self.source = source
        self.sink = sink

    def __eq__(self, other):
        # If the timestamps are equal in `self` (ie, an inplace dependence) then
        # they must be equal in `other` too
        return (self.source == other.source and
                self.sink == other.sink and
                ((self.source.timestamp == self.sink.timestamp) ==
                 (other.source.timestamp == other.sink.timestamp)))

    @property
    def function(self):
        return self.source.function

    @property
    def findices(self):
        return self.source.findices

    @property
    def aindices(self):
        return tuple({i, j} for i, j in zip(self.source.aindices, self.sink.aindices))

    @cached_property
    def distance(self):
        return self.source.distance(self.sink)

    @cached_property
    def _defined_findices(self):
        return frozenset(flatten(i._defines for i in self.findices))

    @cached_property
    def distance_mapper(self):
        return {i: j for i, j in zip(self.findices, self.distance)}

    @cached_property
    def cause(self):
        """Return the findex causing the dependence."""
        for i, j in zip(self.findices, self.distance):
            try:
                if j > 0:
                    return i._defines
            except TypeError:
                # Conservatively assume this is an offending dimension
                return i._defines
        return frozenset()

    @cached_property
    def read(self):
        if self.is_flow:
            return self.sink
        elif self.is_anti:
            return self.source
        else:
            return None

    @cached_property
    def write(self):
        if self.is_flow:
            return self.source
        elif self.is_anti:
            return self.sink
        else:
            return None

    @cached_property
    def is_flow(self):
        return self.source.is_write and self.sink.is_read

    @cached_property
    def is_anti(self):
        return self.source.is_read and self.sink.is_write

    @cached_property
    def is_waw(self):
        return self.source.is_write and self.sink.is_write

    @cached_property
    def is_regular(self):
        return self.source.is_regular and self.sink.is_regular

    @cached_property
    def is_increment(self):
        return self.source.is_increment and self.sink.is_increment

    @cached_property
    def is_irregular(self):
        return not self.is_regular

    @memoized_meth
    def is_carried(self, dim=None):
        """Return True if definitely a dimension-carried dependence, False otherwise."""
        try:
            if dim is None:
                return self.distance > 0
            else:
                return len(self.cause & dim._defines) > 0
        except TypeError:
            # Conservatively assume this is a carried dependence
            return True

    @memoized_meth
    def is_reduce(self, dim):
        """
        Return True if ``dim`` may represent a reduction dimension for
        ``self``, False otherwise.
        """
        test0 = self.is_increment
        test1 = self.is_regular
        test2 = all(i not in self._defined_findices for i in dim._defines)
        return test0 and test1 and test2

    @memoized_meth
    def is_reduce_atmost(self, dim=None):
        """
        More relaxed than :meth:`is_reduce`. Return True  if ``dim`` may
        represent a reduction dimension for ``self`` or if `self`` is definitely
        independent of ``dim``, False otherwise.
        """
        return self.is_reduce(dim) or self.is_indep(dim)

    @memoized_meth
    def is_indep(self, dim=None):
        """
        Return True if definitely a dimension-independent dependence, False otherwise.
        """
        try:
            if self.source.is_irregular or self.sink.is_irregular:
                # Note: we cannot just return `self.distance == 0` as an irregular
                # source/sink might mean that an array is actually accessed indirectly
                # (e.g., A[B[i]]), thus there would be no guarantee on independence
                return False
            elif dim is None:
                return self.distance == 0
            elif self.source.is_local and self.sink.is_local:
                # A dependence between two locally declared scalars
                return True
            else:
                # Note: below, `i in self._defined_findices` is to check whether `i`
                # is actually (one of) the reduction dimension(s), in which case
                # `self` would indeed be a dimension-dependent dependence
                test0 = (not self.is_increment or
                         any(i in self._defined_findices for i in dim._defines))
                test1 = len(self.cause & dim._defines) == 0
                return test0 and test1
        except TypeError:
            # Conservatively assume this is not dimension-independent
            return False

    @memoized_meth
    def is_inplace(self, dim=None):
        """Stronger than ``is_indep()``, as it also compares the timestamps."""
        return self.source.lex_eq(self.sink) and self.is_indep(dim)

    def __repr__(self):
        return "%s -> %s" % (self.source, self.sink)


class DependenceGroup(list):

    @cached_property
    def cause(self):
        return frozenset().union(*[i.cause for i in self])

    @cached_property
    def functions(self):
        """Return the DiscreteFunctions inducing a dependence."""
        return frozenset({i.function for i in self})

    @cached_property
    def none(self):
        return len(self) == 0

    @cached_property
    def increment(self):
        """Return the increment-induced dependences."""
        return DependenceGroup(i for i in self if i.is_increment)

    def carried(self, dim=None):
        """Return the dimension-carried dependences."""
        return DependenceGroup(i for i in self if i.is_carried(dim))

    def independent(self, dim=None):
        """Return the dimension-independent dependences."""
        return DependenceGroup(i for i in self if i.is_indep(dim))

    def inplace(self, dim=None):
        """Return the in-place dependences."""
        return DependenceGroup(i for i in self if i.is_inplace(dim))

    def __add__(self, other):
        assert isinstance(other, DependenceGroup)
        return DependenceGroup(super(DependenceGroup, self).__add__(other))

    def __sub__(self, other):
        assert isinstance(other, DependenceGroup)
        return DependenceGroup([i for i in self if i not in other])

    def project(self, function):
        """
        Return a new DependenceGroup retaining only the dependences due to
        the provided function.
        """
        return DependenceGroup(i for i in self if i.function is function)


class Scope(object):

    def __init__(self, exprs):
        """
        A Scope represents a group of TimedAcces objects extracted
        from some IREq ``exprs``. The expressions must be provided
        in program order.
        """
        exprs = as_tuple(exprs)

        self.reads = {}
        self.writes = {}

        for i, e in enumerate(exprs):
            # Reads
            for j in retrieve_terminals(e.rhs):
                v = self.reads.setdefault(j.function, [])
                mode = 'RI' if e.is_Increment and j.function is e.lhs.function else 'R'
                v.append(TimedAccess(j, mode, i, e.ispace.directions))

            # Write
            v = self.writes.setdefault(e.lhs.function, [])
            mode = 'WI' if e.is_Increment else 'W'
            v.append(TimedAccess(e.lhs, mode, i, e.ispace.directions))

            # If an increment, we got one implicit read
            if e.is_Increment:
                v = self.reads.setdefault(e.lhs.function, [])
                v.append(TimedAccess(e.lhs, 'RI', i, e.ispace.directions))

        # The iterators read symbols too
        dimensions = set().union(*[e.dimensions for e in exprs])
        for d in dimensions:
            for j in d.symbolic_size.free_symbols:
                v = self.reads.setdefault(j.function, [])
                v.append(TimedAccess(j, 'R', -1, {}))

    def getreads(self, function):
        return as_tuple(self.reads.get(function))

    def getwrites(self, function):
        return as_tuple(self.writes.get(function))

    def __getitem__(self, function):
        return self.getwrites(function) + self.getreads(function)

    def __repr__(self):
        tracked = filter_sorted(set(self.reads) | set(self.writes), key=lambda i: i.name)
        maxlen = max(1, max([len(i.name) for i in tracked]))
        out = "{:>%d} =>  W : {}\n{:>%d}     R : {}" % (maxlen, maxlen)
        pad = " "*(maxlen + 9)
        reads = [self.getreads(i) for i in tracked]
        for i, r in enumerate(list(reads)):
            if not r:
                reads[i] = ''
                continue
            first = "%s" % tuple.__repr__(r[0])
            shifted = "\n".join("%s%s" % (pad, tuple.__repr__(j)) for j in r[1:])
            shifted = "%s%s" % ("\n" if shifted else "", shifted)
            reads[i] = first + shifted
        writes = [self.getwrites(i) for i in tracked]
        for i, w in enumerate(list(writes)):
            if not w:
                writes[i] = ''
                continue
            first = "%s" % tuple.__repr__(w[0])
            shifted = "\n".join("%s%s" % (pad, tuple.__repr__(j)) for j in w[1:])
            shifted = "%s%s" % ("\n" if shifted else "", shifted)
            writes[i] = '\033[1;37;31m%s\033[0m' % (first + shifted)
        return "\n".join([out.format(i.name, w, '', r)
                          for i, r, w in zip(tracked, reads, writes)])

    @cached_property
    def accesses(self):
        groups = list(self.reads.values()) + list(self.writes.values())
        return [i for group in groups for i in group]

    @cached_property
    def functions(self):
        return set(self.reads) | set(self.writes)

    @cached_property
    def d_flow(self):
        """Generate all flow (or "read-after-write") dependences."""
        found = DependenceGroup()
        for k, v in self.writes.items():
            for w in v:
                for r in self.reads.get(k, []):
                    try:
                        distance = r.distance(w)
                        is_flow = distance < 0 or (distance == 0 and r.lex_ge(w))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence, unless
                        # it's a read-for-increment
                        is_flow = not r.is_read_increment
                    if is_flow:
                        found.append(Dependence(w, r))
        return found

    @cached_property
    def d_anti(self):
        """Generate all anti (or "write-after-read") dependences."""
        found = DependenceGroup()
        for k, v in self.writes.items():
            for w in v:
                for r in self.reads.get(k, []):
                    try:
                        distance = r.distance(w)
                        is_anti = distance > 0 or (distance == 0 and r.lex_lt(w))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence, unless
                        # it's a read-for-increment
                        is_anti = not r.is_read_increment
                    if is_anti:
                        found.append(Dependence(r, w))
        return found

    @cached_property
    def d_output(self):
        """Generate all output (or "write-after-write") dependences."""
        found = DependenceGroup()
        for k, v in self.writes.items():
            for w1 in v:
                for w2 in self.writes.get(k, []):
                    try:
                        distance = w2.distance(w1)
                        is_output = distance > 0 or (distance == 0 and w2.lex_gt(w1))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_output = True
                    if is_output:
                        found.append(Dependence(w2, w1))
        return found

    @cached_property
    def d_all(self):
        """Generate all flow, anti, and output dependences."""
        return self.d_flow + self.d_anti + self.d_output

    @memoized_meth
    def d_from_access(self, access):
        """Generate all dependences involving a given TimedAccess."""
        return DependenceGroup(d for d in self.d_all
                               if d.source is access or d.sink is access)

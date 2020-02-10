from itertools import chain

from cached_property import cached_property
from sympy import S

from devito.ir.support.space import Backward, IterationSpace
from devito.ir.support.vector import LabeledVector, Vector
from devito.symbolics import retrieve_terminals, q_monoaffine
from devito.tools import (EnrichedTuple, Tag, as_tuple, is_integer,
                          filter_sorted, flatten, memoized_meth, memoized_generator)
from devito.types import Dimension

__all__ = ['IterationInstance', 'TimedAccess', 'Scope']


class IndexMode(Tag):
    """Tag for access functions."""
    pass
AFFINE = IndexMode('affine')  # noqa
IRREGULAR = IndexMode('irregular')


class IterationInstance(LabeledVector):

    """
    A representation of the iteration and data points accessed by an
    Indexed object. Three different concepts are distinguished:

        * Index functions: the expressions describing what *iteration* space
          points are accessed.
        * ``aindices``: the Dimensions acting as iteration variables.
          There is one aindex for each non-constant affine index function. If
          the index function is non-affine, then it may not be possible to detect
          its aindex; in such a case, None is used as placeholder.
        * ``findices``: the Dimensions describing what *data* space point
          are accessed.

    An IterationInstance may be regular or irregular. It is regular if and only
    if *all* index functions are affine in their respective findex.  The
    downside of irregular IterationInstances is that dependence testing is
    harder, which in turn may require the data dependence analyzer to act more
    conservatively.

    Examples
    --------
    Given:
        x, y, z : findices
        w : a generic Dimension

           | x+1 |           |  x  |          |  x  |          | w |          | x+y |
    obj1 = | y+2 | ,  obj2 = |  4  | , obj3 = |  x  | , obj4 = | y | , obj5 = |  y  |
           | z-3 |           | z+1 |          |  y  |          | z |          |  z  |

    We have that:

        * obj1 and obj2 are regular;
        * obj3 is irregular because an findex, ``x``, appears outside of its index
          function (i.e., in the second slot, when ``y`` is expected);
        * obj4 is irregular, because a different dimension, ``w``, is used in place
          of ``x`` within the first index function, where ``x`` is expected;
        * obj5 is irregular, as two findices appear in the same index function.
    """

    def __new__(cls, indexed):
        findices = tuple(indexed.function.dimensions)
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
                    if fi in candidate._defines:
                        if q_monoaffine(i, candidate, self.findices):
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
            elif isinstance(i, Dimension):
                aindices.append(i)
            else:
                dims = {i for i in i.free_symbols if isinstance(i, Dimension)}
                aindices.append(dims.pop() if len(dims) == 1 else None)
        return EnrichedTuple(*aindices, getters=self.findices)

    @property
    def findices(self):
        return self.labels

    @cached_property
    def findices_affine(self):
        return tuple(fi for fi, im in zip(self.findices, self.index_mode) if im is AFFINE)

    @cached_property
    def findices_irregular(self):
        return tuple(fi for fi, im in zip(self.findices, self.index_mode)
                     if im is IRREGULAR)

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

    def distance(self, other):
        """
        Compute the distance from ``self`` to ``other``.

        Parameters
        ----------
        other : IterationInstance
            The IterationInstance from which the distance is computed.
        """
        assert isinstance(other, IterationInstance)
        if self.findices != other.findices:
            raise TypeError("Cannot compute distance due to mismatching `findices`")

        return super(IterationInstance, self).distance(other)


class TimedAccess(IterationInstance):

    """
    An IterationInstance enriched with additional information:

        * a "timestamp"; that is, an integer indicating the statement within
          which the TimedAccess appears in the execution flow;
        * an array of Intervals, which represent the space in which the
          TimedAccess iterates;
        * an array of IterationDirections (one for each findex).

    Notes
    -----
    The comparison operators ``==, !=, <, <=, >, >=`` should be regarded as
    operators for lexicographic ordering of TimedAccess objects, based
    on the values of the index functions and the access mode (read, write).
    """

    _modes = ('R', 'W', 'RI', 'WI')

    def __new__(cls, indexed, mode, timestamp, ispace=None):
        assert mode in cls._modes
        assert is_integer(timestamp)

        obj = super(TimedAccess, cls).__new__(cls, indexed)

        obj.indexed = indexed
        obj.function = indexed.function
        obj.mode = mode
        obj.timestamp = timestamp

        obj.ispace = ispace or IterationSpace([])

        return obj

    def __repr__(self):
        mode = '\033[1;37;31mW\033[0m' if self.is_write else '\033[1;37;32mR\033[0m'
        return "%s<%s,[%s]>" % (mode, self.name, ', '.join(str(i) for i in self))

    def __eq__(self, other):
        return (isinstance(other, TimedAccess) and
                self.function is other.function and
                self.mode == other.mode and
                self.ispace == other.ispace and
                super(TimedAccess, self).__eq__(other))

    def __hash__(self):
        return super(TimedAccess, self).__hash__()

    @property
    def name(self):
        return self.function.name

    @property
    def intervals(self):
        return self.ispace.intervals

    @property
    def directions(self):
        return self.ispace.directions

    @property
    def itintervals(self):
        return self.ispace.itintervals

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

    @cached_property
    def is_regular(self):
        if not super(TimedAccess, self).is_regular:
            return False

        # The order of the `aindices` must match the order of the iteration
        # space Dimensions
        positions = []
        for d in self.aindices:
            for n, i in enumerate(self.intervals):
                if i.dim._defines & d._defines:
                    positions.append(n)
                    break
        return positions == sorted(positions)

    def __lt__(self, other):
        if not isinstance(other, TimedAccess):
            raise TypeError("Cannot compare with object of type %s" % type(other))
        if self.directions != other.directions:
            raise TypeError("Cannot compare due to mismatching `direction`")
        if self.intervals != other.intervals:
            raise TypeError("Cannot compare due to mismatching `intervals`")
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
        """
        Compute the distance from ``self`` to ``other``.

        Parameters
        ----------
        other : TimedAccess
            The TimedAccess from which the distance is computed.
        findex : Dimension, optional
            If supplied, compute the distance only up to and including ``findex``.
        """
        assert isinstance(other, TimedAccess)

        if not self.rank:
            return Vector()

        # Compute distance up to `limit`, ignoring `directions` for the moment
        if findex is None:
            findex = self.findices[-1]
            limit = self.rank
        else:
            try:
                limit = self._cached_findices_index[findex] + 1
            except KeyError:
                raise TypeError("Cannot compute distance as `findex` not in `findices`")
        distance = list(super(TimedAccess, self).distance(other)[:limit])

        # * If mismatching `directions`, set the distance to infinity
        # * If direction is Backward, flip the sign
        ret = []
        for i, it0, it1 in zip(distance, self.itintervals, other.itintervals):
            if it0.direction is it1.direction and it0.interval == it1.interval:
                ret.append(-i if it0.direction is Backward else i)
            else:
                ret.append(S.Infinity)
                break

        return Vector(*ret)

    def touched_halo(self, findex):
        """
        Return a boolean 2-tuple, one entry for each ``findex`` DataSide. True
        means that the halo is touched along that DataSide.
        """
        # If an irregularly (non-affine) accessed Dimension, conservatively
        # assume the halo will be touched
        if self.irregular(findex):
            return (True, True)

        d = self.aindices[findex]

        # If the iterator is *not* a distributed Dimension, then surely the halo
        # isn't touched
        try:
            if not d._maybe_distributed:
                return (False, False)
        except AttributeError:
            pass

        # If a constant (integer, symbolic expr) is used to index into `findex`,
        # there is actually nothing we can do -- the most likely scenario is that
        # it's accessing into a *local* SubDomain/SubDimension
        # TODO: make sure this is indeed the case
        if is_integer(self[findex]) or d not in self[findex].free_symbols:
            return (False, False)

        # Given `d`'s iteration Interval `d[m, M]`, we know that `d` iterates between
        # `d_m + m` and `d_M + M`
        try:
            m, M = self.intervals[d].offsets
        except AttributeError:
            if d.is_NonlinearDerived:
                # We should only end up here with subsampled Dimensions
                m, M = self.intervals[d.root].offsets
            else:
                assert False

        # If `m + (self[d] - d) < self.function._size_nodomain[d].left`, then `self`
        # will definitely touch the left-halo, at least when `d=0`
        size_nodomain_left = self.function._size_nodomain[findex].left
        try:
            touch_halo_left = bool(m + (self[findex] - d) < size_nodomain_left)
        except TypeError:
            # Two reasons we might end up here:
            # * `d` is a constant integer
            # * `m` is a symbol (e.g., a SubDimension-induced offset)
            #   TODO: we could exploit the properties attached to `m` (if any), such
            #         as `nonnegative` etc, to do something smarter than just
            #         assuming, conservatively, `touch_halo_left = True`
            touch_halo_left = True

        # If `M + (self[d] - d) > self.function._size_nodomain[d].left`, then
        # `self` will definitely touch the right-halo, at least when `d=d_M`
        try:
            touch_halo_right = bool(M + (self[findex] - d) > size_nodomain_left)
        except TypeError:
            # See comments in the except block above
            touch_halo_right = True

        return (touch_halo_left, touch_halo_right)


class Dependence(object):

    """
    A data dependence between two TimedAccess objects.
    """

    def __init__(self, source, sink):
        assert isinstance(source, TimedAccess) and isinstance(sink, TimedAccess)
        assert source.function is sink.function
        self.source = source
        self.sink = sink

    def __eq__(self, other):
        # If the timestamps are equal in `self` (ie, an inplace dependence) then
        # they must be equal in `other` too
        return (self.source == other.source and
                self.sink == other.sink and
                ((self.source.timestamp == self.sink.timestamp) ==
                 (other.source.timestamp == other.sink.timestamp)))

    def __hash__(self):
        return hash(
            (self.source, self.sink, self.source.timestamp == self.sink.timestamp)
        )

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
        retval = {}
        for i, j in zip(self.findices, self.distance):
            for d in i._defines:
                retval[d] = j
        return retval

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
    def is_iaw(self):
        """Is it an increment-after-write dependence ?"""
        return self.source.is_write and self.sink.is_increment

    @cached_property
    def is_increment(self):
        return self.source.is_increment and self.sink.is_increment

    @cached_property
    def is_regular(self):
        # Note: what we do below is stronger than something along the lines of
        # `self.source.is_regular and self.sink.is_regular`
        # `source` and `sink` may be regular in isolation, but the dependence
        # itself could be irregular, as the two TimedAccesses may stem from
        # different iteration spaces. Instead if the distance is an integer
        # vector, it is guaranteed that the iteration space is the same
        return all(is_integer(i) for i in self.distance)

    @cached_property
    def is_irregular(self):
        return not self.is_regular

    @cached_property
    def is_lex_positive(self):
        """True if the source preceeds the sink, False otherwise."""
        return self.source.timestamp < self.sink.timestamp

    @cached_property
    def is_lex_equal(self):
        """True if the source has same timestamp as the sink, False otherwise."""
        return self.source.timestamp == self.sink.timestamp

    @cached_property
    def is_lex_negative(self):
        """True if the sink preceeds the source, False otherwise."""
        return self.source.timestamp > self.sink.timestamp

    @cached_property
    def is_lex_non_stmt(self):
        """
        True if either the source or the sink are from non-statements, False otherwise.
        """
        return self.source.timestamp == -1 or self.sink.timestamp == -1

    @property
    def is_local(self):
        return self.function.is_Symbol

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

    @memoized_meth
    def is_storage_related(self, dims=None):
        """
        True if a storage-related dependence, that is multiple iterations
        cause the access of the same memory location, False otherwise.
        """
        for d in self.findices:
            if (d._defines & set(as_tuple(dims)) and
                    any(i.is_NonlinearDerived for i in d._defines)):
                return True
        return False

    def __repr__(self):
        return "%s -> %s" % (self.source, self.sink)


class DependenceGroup(set):

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
        return DependenceGroup(super(DependenceGroup, self).__or__(other))

    def __sub__(self, other):
        assert isinstance(other, DependenceGroup)
        return DependenceGroup(super(DependenceGroup, self).__sub__(other))

    def project(self, function):
        """
        Return a new DependenceGroup retaining only the dependences due to
        the provided function.
        """
        return DependenceGroup(i for i in self if i.function is function)


class Scope(object):

    def __init__(self, exprs):
        """
        A Scope enables data dependence analysis on a totally ordered sequence
        of expressions.
        """
        exprs = as_tuple(exprs)

        self.reads = {}
        self.writes = {}

        self.initialized = set()

        for i, e in enumerate(exprs):
            # Reads
            for j in retrieve_terminals(e.rhs):
                v = self.reads.setdefault(j.function, [])
                mode = 'RI' if e.is_Increment and j.function is e.lhs.function else 'R'
                v.append(TimedAccess(j, mode, i, e.ispace))

            # Write
            v = self.writes.setdefault(e.lhs.function, [])
            mode = 'WI' if e.is_Increment else 'W'
            v.append(TimedAccess(e.lhs, mode, i, e.ispace))

            # If an increment, we got one implicit read
            if e.is_Increment:
                v = self.reads.setdefault(e.lhs.function, [])
                v.append(TimedAccess(e.lhs, 'RI', i, e.ispace))

            # If writing to a scalar, we have an initialization
            if not e.is_Increment and e.is_scalar:
                self.initialized.add(e.lhs.function)

        # The iterators symbols too
        dimensions = set().union(*[e.dimensions for e in exprs])
        for d in dimensions:
            for j in d.symbolic_size.free_symbols:
                v = self.reads.setdefault(j.function, [])
                v.append(TimedAccess(j, 'R', -1))

        # Factor in conditionals
        conditionals = set().union(*[e.conditionals for e in exprs])
        for d in conditionals:
            for j in d.free_symbols:
                v = self.reads.setdefault(j.function, [])
                v.append(TimedAccess(j, 'R', -1))

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

    @memoized_meth
    def a_query(self, timestamps=None, modes=None):
        timestamps = as_tuple(timestamps)
        modes = as_tuple(modes) or TimedAccess._modes
        return tuple(a for a in self.accesses
                     if a.timestamp in timestamps and a.mode in modes)

    @memoized_generator
    def d_flow_gen(self):
        """Generate the flow (or "read-after-write") dependences."""
        for k, v in self.writes.items():
            for w in v:
                for r in self.reads.get(k, []):
                    dependence = Dependence(w, r)
                    distance = dependence.distance
                    try:
                        is_flow = distance > 0 or (r.lex_ge(w) and distance == 0)
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence, unless
                        # it's a read-for-increment
                        is_flow = not r.is_read_increment
                    if is_flow:
                        yield dependence

    @cached_property
    def d_flow(self):
        """Flow (or "read-after-write") dependences."""
        return DependenceGroup(self.d_flow_gen())

    @memoized_generator
    def d_anti_gen(self):
        """Generate the anti (or "write-after-read") dependences."""
        for k, v in self.writes.items():
            for w in v:
                for r in self.reads.get(k, []):
                    dependence = Dependence(r, w)
                    distance = dependence.distance
                    try:
                        is_anti = distance > 0 or (r.lex_lt(w) and distance == 0)
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence, unless
                        # it's a read-for-increment
                        is_anti = not r.is_read_increment
                    if is_anti:
                        yield dependence

    @cached_property
    def d_anti(self):
        """Anti (or "write-after-read") dependences."""
        return DependenceGroup(self.d_anti_gen())

    @memoized_generator
    def d_output_gen(self):
        """Generate the output (or "write-after-write") dependences."""
        for k, v in self.writes.items():
            for w1 in v:
                for w2 in self.writes.get(k, []):
                    dependence = Dependence(w2, w1)
                    distance = dependence.distance
                    try:
                        is_output = distance > 0 or (w2.lex_gt(w1) and distance == 0)
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_output = True
                    if is_output:
                        yield dependence

    @cached_property
    def d_output(self):
        """Output (or "write-after-write") dependences."""
        return DependenceGroup(self.d_output_gen())

    def d_all_gen(self):
        """Generate all flow, anti and output dependences."""
        return chain(self.d_flow_gen(), self.d_anti_gen(), self.d_output_gen())

    @cached_property
    def d_all(self):
        """All flow, anti, and output dependences."""
        return self.d_flow + self.d_anti + self.d_output

    @memoized_generator
    def d_from_access_gen(self, accesses):
        """
        Generate all flow, anti, and output dependences involving any of
        the given TimedAccess objects.
        """
        accesses = as_tuple(accesses)
        for d in self.d_all_gen():
            for i in accesses:
                if d.source is i or d.sink is i:
                    yield d
                    break

    @memoized_meth
    def d_from_access(self, accesses):
        """
        All flow, anti, and output dependences involving any of the given
        TimedAccess objects.
        """
        return DependenceGroup(self.d_from_access_gen(accesses))

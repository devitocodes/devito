from cached_property import cached_property
from sympy import Basic, S

from devito.dimension import Dimension
from devito.ir.support.space import Any, Backward
from devito.symbolics import retrieve_terminals, q_affine, q_inc
from devito.tools import as_tuple, is_integer, filter_sorted

__all__ = ['Vector', 'IterationInstance', 'Access', 'TimedAccess', 'Scope']


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
        if self.rank == 0:
            return 'NullVector'
        maxlen = max(3, max([len(str(i)) for i in self]))
        return '\n'.join([('|{:^%d}|' % maxlen).format(str(i)) for i in self])

    @property
    def rank(self):
        return len(self)

    @property
    def sum(self):
        return sum(self)

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

        Example
        =======
                 | 3 |           | 1 |               |  2  |
        source = | 2 | ,  sink = | 4 | , distance => | -2  |
                 | 1 |           | 5 |               | -4  |

        There are 2, 2, and 4 points between [3-2], [2-4], and [1-5], respectively.
        """
        return self - other


class IterationInstance(Vector):

    """
    A representation of the iteration and data points accessed by an
    :class:`Indexed` object. Three different concepts are distinguished:

        * Index functions: the expressions telling what *iteration* space point
          is accessed.
        * ``aindices``: the :class:`Dimension`s acting as iteration variables.
          There is one aindex for each index function. If the index function
          is non-affine, then it may not be possible to detect its aindex;
          in such a case, None is used as placeholder.
        * ``findices``: the :class:`Dimension`s telling what *data* space point
          is accessed.
    """

    def __new__(cls, indexed):
        obj = super(IterationInstance, cls).__new__(cls, *indexed.indices)
        # findices
        obj.findices = tuple(indexed.base.function.indices)
        if len(obj.findices) != len(set(obj.findices)):
            raise ValueError("Illegal non-unique `findices`")
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

    def __getitem__(self, index):
        if isinstance(index, (slice, int)):
            return super(IterationInstance, self).__getitem__(index)
        elif index in self.findices:
            return super(IterationInstance, self).__getitem__(self.findices.index(index))
        elif isinstance(index, Dimension):
            return None
        else:
            raise TypeError("IterationInstance indices must be integers, slices, or "
                            "Dimensions, not %s" % type(index))

    @cached_property
    def _cached_findices_index(self):
        # Avoiding to call self.findices.index repeatedly speeds analysis up
        return {fi: i for i, fi in enumerate(self.findices)}

    @cached_property
    def index_mode(self):
        index_mode = []
        for i, fi in zip(self, self.findices):
            if is_integer(i) or q_affine(i, fi):
                index_mode.append('regular')
            else:
                dims = {i for i in i.free_symbols if isinstance(i, Dimension)}
                try:
                    # There's still hope it's regular if a DerivedDimension is used
                    candidate = dims.pop()
                    if candidate.parent == fi and q_affine(i, candidate):
                        index_mode.append('regular')
                        continue
                except (KeyError, AttributeError):
                    pass
                index_mode.append('irregular')
        return tuple(index_mode)

    @cached_property
    def aindices(self):
        aindices = []
        for i, fi in zip(self, self.findices):
            if is_integer(i):
                aindices.append(None)
            elif q_affine(i, fi):
                aindices.append(fi)
            else:
                dims = {i for i in i.free_symbols if isinstance(i, Dimension)}
                aindices.append(dims.pop() if len(dims) == 1 else None)
        return tuple(aindices)

    @property
    def is_regular(self):
        return all(i == 'regular' for i in self.index_mode)

    @property
    def is_irregular(self):
        return not self.is_regular

    def distance(self, other, findex=None, view=None):
        """Compute the distance from ``self`` to ``other``.

        :param other: The :class:`IterationInstance` from which the distance
                      is computed.
        :param findex: (Optional) if supplied, compute the distance only up to
                       and including ``findex`` (defaults to None).
        :param view: (Optional) an iterable of ``findices`` (defaults to None); if
                     supplied, project the distance along these dimensions.
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
        """Return a view of ``self`` in which the slots corresponding to the
        provided ``findices`` have been zeroed."""
        return Vector(*[d if i not in as_tuple(findices) else 0
                        for d, i in zip(self, self.findices)])


class Access(IterationInstance):

    """
    A representation of the access performed by a :class:`Indexed` object
    (a scalar in the degenerate case).

    Notes on Access comparison
    ==========================

    The comparison operators ``==, !=, <, <=, >, >=`` should be regarded as
    operators for lexicographic ordering of :class:`Access` objects, based
    on the values of the index functions (and the index functions only).

    For example, if two Access objects A and B employ the same index functions,
    the operation A == B will return True regardless of whether A and B are
    reads or writes or mixed.
    """

    def __new__(cls, indexed, mode):
        assert mode in ['R', 'W', 'RI', 'WI']
        obj = super(Access, cls).__new__(cls, indexed)
        obj.function = indexed.base.function
        obj.mode = mode
        return obj

    def __eq__(self, other):
        return super(Access, self).__eq__(other) and\
            isinstance(other, Access) and\
            self.function == other.function

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

    def __repr__(self):
        mode = '\033[1;37;31mW\033[0m' if self.is_write else '\033[1;37;32mR\033[0m'
        return "%s<%s,[%s]>" % (mode, self.name, ', '.join(str(i) for i in self))


class TimedAccess(Access):

    """
    A special :class:`Access` object enriched with: ::

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
    ========
    Given:
    findices = [x, y, z]
    w = an object of type Dimension

           | x+1 |           |  x  |           |  x  |          | w |          | x+y |
    obj1 = | y+2 | ,  obj2 = |  4  | , obj3 => |  x  | , obj4 = | y | , obj5 = |  y  |
           | z-3 |           | z+1 |           |  y  |          | z |          |  z  |

    We have that: ::

        * obj1 and obj2 are regular;
        * obj3 is irregular because an findex, ``x``, appears outside of its index
          function (i.e., in the second slot, whew ``y`` is expected);
        * obj4 is irregular, because a different dimension, ``w``, is used in place
          of ``x`` within the first index function, where ``x`` is expected;
        * obj5 is irregular, as two findices appear in the same index function --
          the one in the first slot, where only ``x`` is expected.

    """

    def __new__(cls, indexed, mode, timestamp, directions):
        assert is_integer(timestamp)
        obj = super(TimedAccess, cls).__new__(cls, indexed, mode)
        obj.timestamp = timestamp
        obj.directions = [directions.get(i, Any) for i in obj.findices]
        return obj

    def __eq__(self, other):
        return super(TimedAccess, self).__eq__(other) and\
            isinstance(other, TimedAccess) and\
            self.directions == other.directions

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

    """A data dependence between two :class:`Access` objects."""

    def __init__(self, source, sink):
        assert isinstance(source, TimedAccess) and isinstance(sink, TimedAccess)
        assert source.function == sink.function
        self.source = source
        self.sink = sink
        self.findices = source.findices
        self.function = source.function
        self.distance = source.distance(sink)

    @property
    def cause(self):
        """Return the findex causing the dependence (if any -- return None if
        the dependence is between scalars)."""
        for i, j in zip(self.findices, self.distance):
            try:
                if j > 0:
                    return i
            except TypeError:
                # Conservatively assume this is an offending dimension
                return i

    @property
    def is_indirect(self):
        """Return True if induced by an indirection array (e.g., A[B[i]]),
        False otherwise."""
        for d, i, j in zip(self.findices, self.source.index_mode, self.sink.index_mode):
            if d == self.cause and (i == 'irregular' or j == 'irregular'):
                return True
        return False

    @property
    def is_direct(self):
        """Return True if the dependence occurs through affine functions,
        False otherwise."""
        return not self.is_indirect

    @property
    def is_increment(self):
        return self.source.is_increment and self.sink.is_increment

    def is_carried(self, dim=None):
        """Return True if a dimension-carried dependence, False otherwise."""
        try:
            if dim is None:
                return self.distance > 0
            else:
                return any(i == self.cause for i in dim._defines)
        except TypeError:
            # Conservatively assume this is a carried dependence
            return True

    def is_independent(self, dim=None):
        """Return True if a dimension-independent dependence, False otherwise."""
        try:
            if dim is None or self.source.is_irregular or self.sink.is_irregular:
                return self.distance == 0
            else:
                return all(i != self.cause for i in dim._defines)
        except TypeError:
            # Conservatively assume this is not dimension-independent
            return False

    def is_inplace(self, dim=None):
        """Stronger than ``is_independent()``, as it also compares the timestamps."""
        return self.is_independent(dim) and self.source.lex_eq(self.sink)

    def __repr__(self):
        return "%s -> %s" % (self.source, self.sink)


class DependenceGroup(list):

    @property
    def cause(self):
        ret = [i.cause for i in self if i.cause is not None]
        ret.extend([i.parent for i in ret if i.is_Derived])
        return tuple(filter_sorted(ret, key=lambda i: i.name))

    @property
    def none(self):
        return len(self) == 0

    @property
    def direct(self):
        """Return the dependences induced through affine index functions."""
        return DependenceGroup(i for i in self if i.is_direct)

    @property
    def indirect(self):
        """Return the dependences induced through an indirection array."""
        return DependenceGroup(i for i in self if i.is_indirect)

    @property
    def increment(self):
        """Return the increment-induced dependences."""
        return DependenceGroup(i for i in self if i.is_increment)

    def carried(self, dim=None):
        """Return the dimension-carried dependences."""
        return DependenceGroup(i for i in self if i.is_carried(dim))

    def independent(self, dim=None):
        """Return the dimension-independent dependences."""
        return DependenceGroup(i for i in self if i.is_independent(dim))

    def inplace(self, dim=None):
        """Return the in-place dependences."""
        return DependenceGroup(i for i in self if i.is_inplace(dim))

    def __add__(self, other):
        assert isinstance(other, DependenceGroup)
        return DependenceGroup(super(DependenceGroup, self).__add__(other))

    def __sub__(self, other):
        assert isinstance(other, DependenceGroup)
        return DependenceGroup([i for i in self if i not in other])


class Scope(object):

    def __init__(self, exprs):
        """
        A Scope represents a group of :class:`TimedAccess` objects extracted
        from some :class:`IREq` ``exprs``. The expressions must be provided
        in program order.
        """
        exprs = as_tuple(exprs)

        self.reads = {}
        self.writes = {}
        for i, e in enumerate(exprs):
            # reads
            for j in retrieve_terminals(e.rhs):
                v = self.reads.setdefault(j.base.function, [])
                mode = 'R' if not q_inc(e) else 'RI'
                v.append(TimedAccess(j, mode, i, e.ispace.directions))
            # write
            v = self.writes.setdefault(e.lhs.base.function, [])
            mode = 'W' if not q_inc(e) else 'WI'
            v.append(TimedAccess(e.lhs, mode, i, e.ispace.directions))

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
    def has_dep(self):
        """Return True if at least a dependency is detected, False otherwise."""
        for k, v in self.writes.items():
            for w1 in v:
                for r in self.reads.get(k, []):
                    try:
                        is_flow = (r < w1) or (r == w1 and r.lex_ge(w1))
                        is_anti = (r > w1) or (r == w1 and r.lex_lt(w1))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_flow = is_anti = True
                    if is_flow or is_anti:
                        return True
                for w2 in self.writes.get(k, []):
                    try:
                        is_output = (w2 > w1) or (w2 == w1 and w2.lex_gt(w1))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_output = True
                    if is_output:
                        return True
        return False

    @cached_property
    def d_flow(self):
        """Retrieve the flow dependencies, or true dependencies, or read-after-write."""
        found = DependenceGroup()
        for k, v in self.writes.items():
            for w in v:
                for r in self.reads.get(k, []):
                    try:
                        is_flow = (r < w) or (r == w and r.lex_ge(w))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_flow = True
                    if is_flow:
                        found.append(Dependence(w, r))
        return found

    @cached_property
    def d_anti(self):
        """Retrieve the anti dependencies, or write-after-read."""
        found = DependenceGroup()
        for k, v in self.writes.items():
            for w in v:
                for r in self.reads.get(k, []):
                    try:
                        is_anti = (r > w) or (r == w and r.lex_lt(w))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_anti = True
                    if is_anti:
                        found.append(Dependence(r, w))
        return found

    @cached_property
    def d_output(self):
        """Retrieve the output dependencies, or write-after-write."""
        found = DependenceGroup()
        for k, v in self.writes.items():
            for w1 in v:
                for w2 in self.writes.get(k, []):
                    try:
                        is_output = (w2 > w1) or (w2 == w1 and w2.lex_gt(w1))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_output = True
                    if is_output:
                        found.append(Dependence(w2, w1))
        return found

    @cached_property
    def d_all(self):
        """Retrieve all flow, anti, and output dependences."""
        return self.d_flow + self.d_anti + self.d_output

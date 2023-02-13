from itertools import chain

from cached_property import cached_property
from sympy import S
from sympy.tensor.indexed import IndexException

from devito.ir.support.space import Backward, IterationSpace
from devito.ir.support.utils import AccessMode
from devito.ir.support.vector import LabeledVector, Vector
from devito.symbolics import (retrieve_terminals, q_constant, q_affine, q_routine,
                              q_terminal)
from devito.tools import (Tag, as_tuple, is_integer, filter_sorted, flatten,
                          memoized_meth, memoized_generator)
from devito.types import Barrier, Dimension, DimensionTuple, Jump

__all__ = ['IterationInstance', 'TimedAccess', 'Scope']


class IndexMode(Tag):
    """Tag for access functions."""
    pass
AFFINE = IndexMode('affine')  # noqa
REGULAR = IndexMode('regular')
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

    def __new__(cls, access):
        try:
            findices = tuple(access.function.dimensions)
        except AttributeError:
            # E.g., Objects, which don't have `dimensions`
            findices = ()
        if len(findices) != len(set(findices)):
            raise ValueError("Illegal non-unique `findices`")
        try:
            indices = access.indices
        except AttributeError:
            # E.g., `access` is a FieldFromComposite rather than an Indexed
            indices = (S.Infinity,)*len(findices)
        return super().__new__(cls, list(zip(findices, indices)))

    def __hash__(self):
        return super().__hash__()

    @cached_property
    def index_mode(self):
        retval = []
        for i, fi in zip(self, self.findices):
            dims = {j for j in i.free_symbols if isinstance(j, Dimension)}
            if len(dims) == 0 and q_constant(i):
                retval.append(AFFINE)
                continue

            # TODO: Exploit AffineIndexAccessFunction instead of calling
            # q_affine -- ultimately it should get quicker!

            sdims = {d for d in dims if d.is_Stencil}
            if dims == sdims:
                candidates = sdims
            else:
                # E.g. `x + i0 + i1` -> `candidates = {x}`
                candidates = dims - sdims

            if len(candidates) == 1:
                candidate = candidates.pop()
                if fi._defines & candidate._defines:
                    if q_affine(i, candidate):
                        retval.append(AFFINE)
                    else:
                        retval.append(REGULAR)
                else:
                    retval.append(IRREGULAR)
            else:
                retval.append(IRREGULAR)
        return DimensionTuple(*retval, getters=self.findices)

    @cached_property
    def aindices(self):
        retval = []
        for i, fi in zip(self, self.findices):
            dims = {j for j in i.free_symbols if isinstance(j, Dimension)}
            sdims = {d for d in dims if d.is_Stencil}
            candidates = dims - sdims

            if len(candidates) == 1:
                retval.append(candidates.pop())
            elif isinstance(i, Dimension):
                retval.append(i)
            else:
                retval.append(None)
        return DimensionTuple(*retval, getters=self.findices)

    @property
    def findices(self):
        return self.labels

    @cached_property
    def defined_findices_affine(self):
        ret = set()
        for fi, im in zip(self.findices, self.index_mode):
            if im is AFFINE:
                ret.update(fi._defines)
        return ret

    @cached_property
    def defined_findices_irregular(self):
        ret = set()
        for fi, im in zip(self.findices, self.index_mode):
            if im is IRREGULAR:
                ret.update(fi._defines)
        return ret

    def affine(self, findices):
        """
        Return True if all of the provided findices appear in self and are
        affine, False otherwise.
        """
        return set(as_tuple(findices)).issubset(self.defined_findices_affine)

    def affine_if_present(self, findices):
        """
        Return False if any of the provided findices appears in self and
        is not affine, True otherwise.
        """
        present_findices = set(as_tuple(findices)) & set(self.findices)
        return present_findices.issubset(self.defined_findices_affine)

    def irregular(self, findices):
        """
        Return True if all of the provided findices appear in self and are
        irregular, False otherwise.
        """
        return set(as_tuple(findices)).issubset(self.defined_findices_irregular)

    @cached_property
    def is_regular(self):
        return all(i in (AFFINE, REGULAR) for i in self.index_mode)

    @property
    def is_irregular(self):
        return not self.is_regular

    @property
    def is_scalar(self):
        return self.rank == 0


class TimedAccess(IterationInstance, AccessMode):

    """
    A TimedAccess ties together an IterationInstance and an AccessMode.

    Further:

        * It may be associated with an IterationSpace.
        * It carries a "timestamp", that is an integer indicating the statement
          within which the TimedAccess appears in the execution flow.

    Notes
    -----
    The comparison operators ``==, !=, <, <=, >, >=`` should be regarded as
    operators for lexicographic ordering of TimedAccess objects, based
    on the values of the index functions and the access mode (read, write).
    """

    def __new__(cls, access, mode, timestamp, ispace=None):
        obj = super().__new__(cls, access)
        AccessMode.__init__(obj, mode=mode)
        return obj

    def __init__(self, access, mode, timestamp, ispace=None):
        assert is_integer(timestamp)

        self.access = access
        self.timestamp = timestamp
        self.ispace = ispace or IterationSpace([])

    def __repr__(self):
        mode = '\033[1;37;31mW\033[0m' if self.is_write else '\033[1;37;32mR\033[0m'
        return "%s<%s,[%s]>" % (mode, self.name, ', '.join(str(i) for i in self))

    def __eq__(self, other):
        if not isinstance(other, TimedAccess):
            return False

        # At this point no need to go through the class hierarchy's __eq__,
        # which might require expensive comparisons of Vector entries (i.e.,
        # SymPy expressions)

        return (self.access is other.access and  # => self.function is other.function
                self.mode == other.mode and
                self.ispace == other.ispace)

    def __hash__(self):
        return super().__hash__()

    @property
    def function(self):
        return self.access.function

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
    def is_local(self):
        return self.function.is_Symbol

    @cached_property
    def is_regular(self):
        if not super().is_regular:
            return False

        # The order of the `aindices` must match the order of the iteration
        # space Dimensions
        positions = []
        for d in self.aindices:
            try:
                for n, i in enumerate(self.intervals):
                    if i.dim._defines & d._defines:
                        positions.append(n)
                        break
            except AttributeError:
                # `d is None` due to e.g. constant access
                continue
        return positions == sorted(positions)

    def __lt__(self, other):
        if not isinstance(other, TimedAccess):
            raise TypeError("Cannot compare with object of type %s" % type(other))
        if self.directions != other.directions:
            raise TypeError("Cannot compare due to mismatching `direction`")
        if self.intervals != other.intervals:
            raise TypeError("Cannot compare due to mismatching `intervals`")
        return super().__lt__(other)

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

    def distance(self, other):
        """
        Compute the distance from ``self`` to ``other``.

        Parameters
        ----------
        other : TimedAccess
            The TimedAccess w.r.t. which the distance is computed.
        """
        ret = []
        for sit, oit in zip(self.itintervals, other.itintervals):
            n = len(ret)

            try:
                sai = self.aindices[n]
                oai = other.aindices[n]
            except IndexError:
                # E.g., `self=R<f,[x]>` and `self.itintervals=(x, i)`
                break

            try:
                if not (sit == oit and sai.root is oai.root):
                    # E.g., `self=R<f,[x + 2]>` and `other=W<f,[i + 1]>`
                    # E.g., `self=R<f,[x]>`, `other=W<f,[x + 1]>`,
                    #       `self.itintervals=(x<0>,)` and `other.itintervals=(x<1>,)`
                    ret.append(S.Infinity)
                    break
            except AttributeError:
                # E.g., `self=R<f,[cy]>` and `self.itintervals=(y,)` => `sai=None`
                pass

            if self.function._mem_shared:
                # Special case: the distance between two regular, thread-shared
                # objects fallbacks to zero, as any other value would be nonsensical.
                ret.append(S.Zero)

            elif sai and oai and sai._defines & sit.dim._defines:
                # E.g., `self=R<f,[t + 1, x]>`, `self.itintervals=(time, x)`
                # and `ai=t`
                if sit.direction is Backward:
                    ret.append(other[n] - self[n])
                else:
                    ret.append(self[n] - other[n])

            elif not sai and not oai:
                # E.g., `self=R<a,[3]>` and `other=W<a,[4]>`
                if self[n] - other[n] == 0:
                    ret.append(S.Zero)
                else:
                    break

            elif sai in self.ispace and oai in other.ispace:
                # E.g., `self=R<f,[x, y]>`, `sai=time`, self.itintervals=(time, x, y)
                # with `n=0`
                continue

            elif any(d and d._defines & sit.dim._defines for d in (sai, oai)):
                # In some cases, the distance degenerates because `self` and
                # `other` never intersect, which essentially means there's no
                # dependence between them. In this case, we set the distance to
                # a dummy value (the imaginary unit). Hence, we call these
                # "imaginary dependences". This occurs in just a small set of
                # special cases, which we handle here

                # Case 1: `sit` is an IterationInterval with statically known
                # trip count. E.g. it ranges from 0 to 3; `other` performs a
                # constant access at 4
                for v in (self[n], other[n]):
                    try:
                        if bool(v < sit.symbolic_min or v > sit.symbolic_max):
                            return Vector(S.ImaginaryUnit)
                    except TypeError:
                        pass

                # Case 2: `sit` is an IterationInterval over a local SubDimension
                # and `other` performs a constant access
                for d0, d1 in ((sai, oai), (oai, sai)):
                    if d0 is None and d1.is_Sub and d1.local:
                        return Vector(S.ImaginaryUnit)

                # Fallback
                ret.append(S.Infinity)
                break

            elif self.findices[n] in sit.dim._defines:
                # E.g., `self=R<u,[t+1, ii_src_0+1, ii_src_1+2]>` and `fi=p_src` (`n=1`)
                ret.append(S.Infinity)
                break

        if S.Infinity in ret:
            return Vector(*ret)

        # It still could be an imaginary dependence, e.g. `a[3] -> a[4]` or, more
        # nasty, `a[i+1, 3] -> a[i, 4]`
        n = len(ret)
        for i, j in zip(self[n:], other[n:]):
            if i == j:
                ret.append(S.Zero)
            else:
                v = i - j
                if v.is_Number and v.is_finite:
                    return Vector(S.ImaginaryUnit)

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
        m, M = self.intervals[d].offsets

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

    def __repr__(self):
        return "%s -> %s" % (self.source, self.sink)

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
        """Is it an reduction-after-write dependence ?"""
        return self.source.is_write and self.sink.is_reduction

    @cached_property
    def is_reduction(self):
        return self.source.is_reduction or self.sink.is_reduction

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

    @property
    def is_imaginary(self):
        return S.ImaginaryUnit in self.distance

    @memoized_meth
    def is_const(self, dim):
        """
        True if a constant dependence, that is no Dimensions involved, False otherwise.
        """
        return (self.source.aindices[dim] is None and
                self.sink.aindices[dim] is None and
                self.distance_mapper[dim] == 0)

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
        return (self.is_reduction and
                self.is_regular and
                not (dim._defines & self._defined_findices))

    @memoized_meth
    def is_reduce_atmost(self, dim=None):
        """
        More relaxed than :meth:`is_reduce`. Return True  if ``dim`` may
        represent a reduction dimension for ``self`` or if `self`` is definitely
        independent of ``dim``, False otherwise.
        """
        return (not (dim._defines & self._defined_findices)
                or self.is_indep(dim))

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
                # The only hope at this point is that the irregularity is not along
                # `dim` *and* that the distance along `dim` is 0 (e.g., dim=x and
                # f[x, g[x, y]] -> f[x, h[x, y]])
                try:
                    return (self.source.affine(dim) and
                            self.sink.affine(dim) and
                            self.distance_mapper[dim] == 0)
                except KeyError:
                    # `dim is None` or anything not in `self._defined_findices`
                    return False
            if dim is None:
                return self.distance == 0
            else:
                # The only hope at this point is that `dim` appears in the findices
                # with distance 0 (that is, it's not the cause of the dependence)
                return (any(i in self._defined_findices for i in dim._defines) and
                        len(self.cause & dim._defines) == 0)
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
            if d._defines & set(as_tuple(dims)):
                if any(i.is_NonlinearDerived for i in d._defines) or \
                   self.is_const(d):
                    return True
        return False


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
    def reduction(self):
        """Return the reduction-induced dependences."""
        return DependenceGroup(i for i in self if i.is_reduction)

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

    def __init__(self, exprs, rules=None):
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
            terminals = retrieve_terminals(e.rhs, deep=True, mode='unique')
            try:
                terminals.update(retrieve_terminals(e.lhs.indices))
            except AttributeError:
                pass
            for j in terminals:
                v = self.reads.setdefault(j.function, [])
                mode = 'RR' if e.is_Reduction and j.function is e.lhs.function else 'R'
                v.append(TimedAccess(j, mode, i, e.ispace))

            # Write
            terminals = []
            if q_terminal(e.lhs):
                terminals.append(e.lhs)
            if q_routine(e.rhs):
                try:
                    terminals.extend(e.rhs.writes)
                except AttributeError:
                    # E.g., foreign routines, such as `cos` or `sin`
                    pass
            for j in terminals:
                v = self.writes.setdefault(j.function, [])
                mode = 'WR' if e.is_Reduction else 'W'
                v.append(TimedAccess(j, mode, i, e.ispace))

            # If a reduction, we got one implicit read
            if e.is_Reduction:
                v = self.reads.setdefault(e.lhs.function, [])
                v.append(TimedAccess(e.lhs, 'RR', i, e.ispace))

            # If writing to a scalar, we have an initialization
            if not e.is_Reduction and e.is_scalar:
                self.initialized.add(e.lhs.function)

            # Look up ConditionalDimensions
            for v in e.conditionals.values():
                for j in retrieve_terminals(v):
                    v = self.reads.setdefault(j.function, [])
                    v.append(TimedAccess(j, 'R', -1, e.ispace))

        # The iteration symbols too
        dimensions = set().union(*[e.dimensions for e in exprs])
        for d in dimensions:
            for i in d.free_symbols | d.bound_symbols:
                v = self.reads.setdefault(i.function, [])
                v.append(TimedAccess(i, 'R', -1))

        # Objects altering the control flow (e.g., synchronization barriers,
        # break statements, ...) are converted into mock dependences
        for i, e in enumerate(exprs):
            if e.find(Barrier) | e.find(Jump):
                for a in self.accesses:
                    f = a.function
                    indices = [i if d in e.ispace else S.Infinity
                               for i, d in zip(a, a.aindices)]
                    try:
                        mock = f.indexify(indices)
                    except IndexException:
                        continue
                    v = self.writes.setdefault(f, [])
                    v.extend([TimedAccess(mock, 'R', i, e.ispace),
                              TimedAccess(mock, 'W', i, e.ispace)])

        # A set of rules to drive the collection of dependencies
        self.rules = as_tuple(rules)
        assert all(callable(i) for i in self.rules)

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
    def indexeds(self):
        return tuple(i.access for i in self.accesses if i.access.is_Indexed)

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

                    if any(not rule(dependence) for rule in self.rules):
                        continue

                    distance = dependence.distance
                    try:
                        is_flow = distance > 0 or (r.lex_ge(w) and distance == 0)
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence, unless
                        # it's a read-for-reduction
                        is_flow = (not dependence.is_imaginary and
                                   not r.is_read_reduction)
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

                    if any(not rule(dependence) for rule in self.rules):
                        continue

                    distance = dependence.distance
                    try:
                        is_anti = distance > 0 or (r.lex_lt(w) and distance == 0)
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence, unless
                        # it's a read-for-reduction
                        is_anti = (not dependence.is_imaginary and
                                   not r.is_read_reduction)
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

                    if any(not rule(dependence) for rule in self.rules):
                        continue

                    distance = dependence.distance
                    try:
                        is_output = distance > 0 or (w2.lex_gt(w1) and distance == 0)
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_output = not dependence.is_imaginary
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

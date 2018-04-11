from sympy import Eq

from devito.dimension import SubDimension
from devito.equation import DOMAIN, INTERIOR
from devito.ir.support import (IterationSpace, DataSpace, Interval, IntervalGroup, Any,
                               detect_accesses, detect_oobs, force_directions, detect_io,
                               build_intervals, detect_flow_directions)
from devito.symbolics import FrozenExpr, dimension_sort

__all__ = ['LoweredEq', 'ClusterizedEq', 'DummyEq']


class IREq(object):

    """
    A mixin providing operations common to all :mod:`ir` equation types.
    """

    @property
    def is_Scalar(self):
        return self.lhs.is_Symbol

    @property
    def is_Tensor(self):
        return self.lhs.is_Indexed

    @property
    def ispace(self):
        return self._ispace

    @property
    def dspace(self):
        return self._dspace

    @property
    def dimensions(self):
        # Note: some dimensions may be in the iteration space but not in the
        # data space (e.g., a DerivedDimension); likewise, some dimensions may
        # be in the data space but not in the iteration space (e.g., when a
        # function is indexed with integers only)
        return set(self.dspace.dimensions) | set(self.ispace.dimensions)

    @property
    def directions(self):
        return self.ispace.directions

    @property
    def dtype(self):
        return self.lhs.dtype


class LoweredEq(Eq, IREq):

    """
    LoweredEq(expr)
    LoweredEq(expr, ispace, dspace, reads, writes)
    LoweredEq(lhs, rhs, stamp=LoweredEq)

    A SymPy equation with associated :class:`IterationSpace` and
    :class:`DataSpace`.

    When created as ``LoweredEq(expr)``, the iteration and data spaces are
    automatically derived from analysis of ``expr``.
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            # origin: LoweredEq(expr)
            expr = input_expr = args[0]
            assert not isinstance(expr, LoweredEq) and isinstance(expr, Eq)
        elif len(args) == 2:
            # origin: LoweredEq(lhs, rhs, stamp=...)
            stamp = kwargs.pop('stamp')
            expr = Eq.__new__(cls, *args, evaluate=False)
            assert isinstance(stamp, Eq)
            expr.is_Increment = stamp.is_Increment
            expr._ispace, expr._dspace = stamp.ispace, stamp.dspace
            expr.reads, expr.writes = stamp.reads, stamp.writes
            return expr
        elif len(args) == 5:
            # origin: LoweredEq(expr, ispace, space)
            input_expr, ispace, dspace, reads, writes = args
            assert isinstance(ispace, IterationSpace) and isinstance(dspace, DataSpace)
            expr = Eq.__new__(cls, *input_expr.args, evaluate=False)
            expr.is_Increment = input_expr.is_Increment
            expr._ispace, expr._dspace = ispace, dspace
            expr.reads, expr.writes = reads, writes
            return expr
        else:
            raise ValueError("Cannot construct LoweredEq from args=%s "
                             "and kwargs=%s" % (str(args), str(kwargs)))

        # Well-defined dimension ordering
        ordering = dimension_sort(expr, key=lambda i: not i.is_Time)

        # Introduce space sub-dimensions if need to
        region = getattr(input_expr, '_region', DOMAIN)
        if region == INTERIOR:
            mapper = {i: SubDimension("%si" % i, i, 1, -1)
                      for i in ordering if i.is_Space}
            expr = expr.xreplace(mapper)
            ordering = [mapper.get(i, i) for i in ordering]

        # Analyze data accesses
        mapper = detect_accesses(expr)
        oobs = detect_oobs(mapper)

        # The iteration space is constructed so that information always flows
        # from an iteration to another (i.e., no anti-dependences are created)
        directions, _ = force_directions(detect_flow_directions(expr), lambda i: Any)
        intervals, iterators = build_intervals(mapper)
        intervals = sorted(intervals, key=lambda i: ordering.index(i.dim))
        ispace = IterationSpace([i.zero() for i in intervals], iterators, directions)

        # The data space is relative to the computational domain
        intervals = [i if i.dim in oobs else i.zero() for i in intervals]
        intervals += [Interval(i, 0, 0) for i in ordering if i not in ispace.dimensions]
        parts = {k: IntervalGroup(Interval(i, min(j), max(j)) for i, j in v.items())
                 for k, v in mapper.items()}
        dspace = DataSpace(intervals, parts)

        # Finally create the LoweredEq with all metadata attached
        expr = super(LoweredEq, cls).__new__(cls, expr.lhs, expr.rhs, evaluate=False)
        expr.is_Increment = getattr(input_expr, 'is_Increment', False)
        expr._dspace = dspace
        expr._ispace = ispace
        expr.reads, expr.writes = detect_io(expr)

        return expr

    def xreplace(self, rules):
        return LoweredEq(self.lhs.xreplace(rules), self.rhs.xreplace(rules), stamp=self)

    def func(self, *args):
        return super(LoweredEq, self).func(*args, stamp=self, evaluate=False)


class ClusterizedEq(Eq, IREq, FrozenExpr):

    """
    ClusterizedEq(expr, ispace, dspace)
    ClusterizedEq(lhs, rhs, stamp=ClusterizedEq)

    A SymPy equation with associated :class:`IterationSpace` and
    :class:`DataSpace`.

    There are two main differences between a :class:`LoweredEq` and a
    ClusterizedEq: ::

        * In a ClusterizedEq, the iteration and data spaces must *always*
          be provided by the caller.
        * A ClusterizedEq is "frozen", meaning that any call to ``xreplace``
          will not trigger re-evaluation (e.g., mathematical simplification)
          of the expression.

    These two properties make a ClusterizedEq suitable for use in a
    :class:`Cluster`.
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 2:
            # origin: ClusterizedEq(lhs, rhs, stamp=...)
            stamp = kwargs.pop('stamp')
            assert isinstance(stamp, ClusterizedEq)
            expr = Eq.__new__(cls, *args, evaluate=False)
            expr.is_Increment = stamp.is_Increment
            expr._ispace, expr._dspace = stamp.ispace, stamp.dspace
        elif len(args) == 3:
            # origin: ClusterizedEq(expr, ispace, dspace)
            input_expr, ispace, dspace = args
            assert isinstance(ispace, IterationSpace) and isinstance(dspace, DataSpace)
            expr = Eq.__new__(cls, *input_expr.args, evaluate=False)
            expr.is_Increment = input_expr.is_Increment
            expr._ispace, expr._dspace = ispace, dspace
        else:
            raise ValueError("Cannot construct ClusterizedEq from args=%s "
                             "and kwargs=%s" % (str(args), str(kwargs)))
        return expr

    def func(self, *args, **kwargs):
        return super(ClusterizedEq, self).func(*args, stamp=self)


class DummyEq(ClusterizedEq):

    """
    DummyEq(expr)
    DummyEq(lhs, rhs)

    A special :class:`ClusterizedEq` that should never be used except for
    writing tests, when there is no need to know the iteration/data spaces.
    """

    def __new__(cls, *args):
        if len(args) == 1:
            input_expr = args[0]
            assert isinstance(input_expr, Eq)
            obj = LoweredEq(input_expr)
        elif len(args) == 2:
            obj = LoweredEq(Eq(*args, evaluate=False))
        else:
            raise ValueError("Cannot construct DummyEq from args=%s" % str(args))
        return ClusterizedEq.__new__(cls, obj, obj.ispace, obj.dspace)

from sympy import Eq

from devito.dimension import SubDimension
from devito.equation import DOMAIN, INTERIOR
from devito.ir.support import (IterationSpace, DataSpace, Interval, IntervalGroup, Any,
                               detect_accesses, detect_oobs, force_directions, detect_io,
                               build_intervals, detect_flow_directions, align_accesses)
from devito.symbolics import FrozenExpr, dimension_sort

__all__ = ['LoweredEq', 'ClusterizedEq', 'IREq']


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
    def directions(self):
        return self.ispace.directions


class LoweredEq(Eq, IREq):

    """
    LoweredEq(expr)

    A SymPy equation with associated :class:`IterationSpace` and
    :class:`DataSpace`.
    """

    def __new__(cls, *args, **kwargs):
        # Parse input
        if len(args) == 1:
            expr = input_expr = args[0]
            assert type(expr) != LoweredEq
            assert isinstance(expr, Eq)
        elif len(args) == 2:
            # Reconstructing from existing Eq. E.g., we end up here after xreplace
            stamp = kwargs.pop('stamp')
            expr = Eq.__new__(cls, *args, evaluate=False)
            assert isinstance(stamp, Eq)
            expr.is_Increment = stamp.is_Increment
            expr.ispace = stamp.ispace
            expr.dspace = stamp.dspace
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

        # Align data accesses to the computational domain
        expr = align_accesses(expr)

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
        expr.dspace = dspace
        expr.ispace = ispace
        expr.dimensions = ordering
        expr.reads, expr.writes = detect_io(expr)

        return expr

    def xreplace(self, rules):
        return LoweredEq(self.lhs.xreplace(rules), self.rhs.xreplace(rules), stamp=self)

    def func(self, *args):
        return super(LoweredEq, self).func(*args, stamp=self, evaluate=False)


class ClusterizedEq(Eq, IREq, FrozenExpr):

    """
    ClusterizedEq(expr, ispace)

    A SymPy equation with associated :class:`IterationSpace` and
    :class:`DataSpace`.

    There are two main differences between a :class:`LoweredEq` and a
    ClusterizedEq: ::

        * In a ClusterizedEq, the iteration and data spaces are provided and
          thus simply attached to the object.
        * A ClusterizedEq is "frozen", meaning that any call to ``xreplace``
          will not trigger re-evaluation (e.g., mathematical simplification)
          of the expression.

    These two properties make a ClusterizedEq suitable for :class:`Cluster`s.
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 3:
            # ClusterizedEq created from ClusterizedEq(...)
            input_expr, ispace, dspace = args
            assert isinstance(ispace, IterationSpace)
            assert isinstance(dspace, DataSpace)
            expr = Eq.__new__(cls, *input_expr.args, evaluate=False)
            expr.is_Increment = input_expr.is_Increment
            expr.dspace = dspace
            expr.ispace = ispace
        else:
            # ClusterizedEq created via .func(...)
            assert len(args) == 2
            expr = Eq.__new__(cls, *args, evaluate=False)
            expr.is_Increment = kwargs.get('is_Increment', False)
            expr.dspace = kwargs['dspace']
            expr.ispace = kwargs['ispace']
        return expr

    def func(self, *args, **kwargs):
        return super(ClusterizedEq, self).func(*args, dspace=self.dspace,
                                               ispace=self.ispace)

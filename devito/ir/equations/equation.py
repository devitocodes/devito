from sympy import Eq

from devito.dimension import SubDimension
from devito.equation import DOMAIN, INTERIOR
from devito.ir.equations.algorithms import dimension_sort
from devito.ir.support import (IterationSpace, DataSpace, Interval, IntervalGroup,
                               Any, Stencil, detect_accesses, detect_oobs, detect_io,
                               force_directions, detect_flow_directions, build_intervals,
                               build_iterators)
from devito.symbolics import FrozenExpr
from devito.tools import Pickable, as_tuple, split

__all__ = ['LoweredEq', 'ClusterizedEq', 'DummyEq']


class IREq(object):

    """
    A mixin providing operations common to all :mod:`ir` equation types.
    """

    _state = ('is_Increment', 'ispace', 'dspace', 'conditionals')

    @property
    def is_Scalar(self):
        return self.lhs.is_Symbol

    @property
    def is_Tensor(self):
        return self.lhs.is_Indexed

    @property
    def is_Increment(self):
        return self._is_Increment

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
    def conditionals(self):
        return as_tuple(self._conditionals)

    @property
    def directions(self):
        return self.ispace.directions

    @property
    def dtype(self):
        return self.lhs.dtype

    @property
    def grid(self):
        return self.lhs.function.grid if self.is_Tensor else None

    @property
    def state(self):
        return {i: getattr(self, i) for i in self._state}


class LoweredEq(Eq, IREq):

    """
    LoweredEq(sympy.Eq)
    LoweredEq(devito.LoweredEq, **kwargs)
    LoweredEq(lhs, rhs, **kwargs)

    A SymPy equation with associated :class:`IterationSpace` and
    :class:`DataSpace`.

    When created as ``LoweredEq(sympy.Eq)``, the iteration and data spaces are
    automatically derived from analysis of ``expr``.

    When created as ``LoweredEq(devito.LoweredEq, **kwargs)``, the keyword
    arguments can be anything that appears in ``LoweredEq._state`` (i.e.,
    ispace, dspace, ...).

    When created as ``LoweredEq(lhs, rhs, **kwargs)``, *all* keywords in
    ``LoweredEq._state`` must appear in ``kwargs``.
    """

    _state = IREq._state + ('reads', 'writes')

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], LoweredEq):
            # origin: LoweredEq(devito.LoweredEq, **kwargs)
            input_expr = args[0]
            expr = Eq.__new__(cls, *input_expr.args, evaluate=False)
            for i in cls._state:
                setattr(expr, '_%s' % i, kwargs.get(i) or getattr(input_expr, i))
            return expr
        elif len(args) == 1 and isinstance(args[0], Eq):
            # origin: LoweredEq(sympy.Eq)
            input_expr = expr = args[0]
        elif len(args) == 2:
            expr = Eq.__new__(cls, *args, evaluate=False)
            for i in cls._state:
                setattr(expr, '_%s' % i, kwargs.pop(i))
            return expr
        else:
            raise ValueError("Cannot construct LoweredEq from args=%s "
                             "and kwargs=%s" % (str(args), str(kwargs)))

        # Well-defined dimension ordering
        dimensions = dimension_sort(expr, key=lambda d: not d.is_Time)
        conditionals, ordering = split(dimensions, lambda d: d.is_Conditional)

        # Introduce space sub-dimensions if need to
        region = getattr(input_expr, '_region', DOMAIN)
        if region == INTERIOR:
            mapper = {i: SubDimension.middle("%si" % i, i, 1, 1)
                      for i in ordering if i.is_Space}
            expr = expr.xreplace(mapper)
            for k, v in mapper.items():
                ordering.insert(ordering.index(k) + 1, v)

        # Analyze the expression
        mapper = detect_accesses(expr)
        oobs = detect_oobs(mapper)

        # The iteration space is constructed so that information always flows
        # from an iteration to another (i.e., no anti-dependences are created)
        directions, _ = force_directions(detect_flow_directions(expr), lambda i: Any)
        iterators = build_iterators(mapper)
        intervals = build_intervals(Stencil.union(*mapper.values()))
        intervals = sorted(intervals, key=lambda i: ordering.index(i.dim))
        ispace = IterationSpace([i.zero() for i in intervals], iterators, directions)

        # The data space is relative to the computational domain
        intervals = [i if i.dim in oobs else i.zero() for i in intervals]
        intervals += [Interval(i, 0, 0) for i in ordering if i not in ispace.dimensions]
        parts = {k: IntervalGroup(build_intervals(v)) for k, v in mapper.items() if k}
        dspace = DataSpace(intervals, parts)

        # Finally create the LoweredEq with all metadata attached
        expr = super(LoweredEq, cls).__new__(cls, expr.lhs, expr.rhs, evaluate=False)
        expr._is_Increment = getattr(input_expr, 'is_Increment', False)
        expr._dspace = dspace
        expr._ispace = ispace
        expr._conditionals = conditionals
        expr._reads, expr._writes = detect_io(expr)

        return expr

    @property
    def reads(self):
        return self._reads

    @property
    def writes(self):
        return self._writes

    def xreplace(self, rules):
        return LoweredEq(self.lhs.xreplace(rules), self.rhs.xreplace(rules), **self.state)

    def func(self, *args):
        return super(LoweredEq, self).func(*args, **self.state, evaluate=False)


class ClusterizedEq(Eq, IREq, FrozenExpr, Pickable):

    """
    ClusterizedEq(devito.IREq, **kwargs)
    ClusterizedEq(lhs, rhs, **kwargs)

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
        if len(args) == 1:
            # origin: ClusterizedEq(expr, **kwargs)
            input_expr = args[0]
            expr = Eq.__new__(cls, *input_expr.args, evaluate=False)
            for i in cls._state:
                v = kwargs[i] if i in kwargs else getattr(input_expr, i, None)
                setattr(expr, '_%s' % i, v)
        elif len(args) == 2:
            # origin: ClusterizedEq(lhs, rhs, **kwargs)
            expr = Eq.__new__(cls, *args, evaluate=False)
            for i in cls._state:
                setattr(expr, '_%s' % i, kwargs.pop(i))
        else:
            raise ValueError("Cannot construct ClusterizedEq from args=%s "
                             "and kwargs=%s" % (str(args), str(kwargs)))
        return expr

    def func(self, *args, **kwargs):
        return super(ClusterizedEq, self).func(*args, **self.state)

    # Pickling support
    _pickle_args = ['lhs', 'rhs']
    _pickle_kwargs = IREq._state
    __reduce_ex__ = Pickable.__reduce_ex__


class DummyEq(ClusterizedEq):

    """
    DummyEq(expr)
    DummyEq(lhs, rhs)

    A special :class:`ClusterizedEq` that tracks no iteration or data spaces.
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
        return ClusterizedEq.__new__(cls, obj, ispace=obj.ispace, dspace=obj.dspace)

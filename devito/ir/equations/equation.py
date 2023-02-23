from cached_property import cached_property
import sympy

from devito.ir.equations.algorithms import dimension_sort, lower_exprs
from devito.finite_differences.differentiable import diff2sympy
from devito.ir.support import (GuardFactor, Interval, IntervalGroup, IterationSpace,
                               Stencil, detect_io, detect_accesses)
from devito.symbolics import IntDiv, uxreplace
from devito.tools import Pickable, Tag, frozendict
from devito.types import Eq, Inc, ReduceMax, ReduceMin

__all__ = ['LoweredEq', 'ClusterizedEq', 'DummyEq', 'OpInc', 'OpMin', 'OpMax']


class IREq(sympy.Eq, Pickable):

    __rargs__ = ('lhs', 'rhs')
    __rkwargs__ = ('ispace', 'conditionals', 'implicit_dims', 'operation')

    @property
    def is_Scalar(self):
        return self.lhs.is_Symbol

    is_scalar = is_Scalar

    @property
    def ispace(self):
        return self._ispace

    @cached_property
    def dimensions(self):
        return set(self.ispace.dimensions)

    @property
    def implicit_dims(self):
        return self._implicit_dims

    @cached_property
    def conditionals(self):
        return self._conditionals or frozendict()

    @property
    def directions(self):
        return self.ispace.directions

    @property
    def dtype(self):
        return self.lhs.dtype

    @property
    def state(self):
        return {i: getattr(self, i) for i in self.__rkwargs__}

    @property
    def operation(self):
        return self._operation

    @property
    def is_Reduction(self):
        return self.operation in (OpInc, OpMin, OpMax)

    @property
    def is_Increment(self):
        return self.operation is OpInc

    def apply(self, func):
        """
        Apply a callable to `self` and each expr-like attribute carried by `self`,
        thus triggering a reconstruction.
        """
        args = [func(self.lhs), func(self.rhs)]
        kwargs = dict(self.state)
        kwargs['conditionals'] = {k: func(v) for k, v in self.conditionals.items()}
        return self.func(*args, **kwargs)

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class Operation(Tag):

    """
    Special operation performed by an Eq.
    """

    @classmethod
    def detect(cls, expr):
        reduction_mapper = {
            Inc: OpInc,
            ReduceMax: OpMax,
            ReduceMin: OpMin
        }
        try:
            return reduction_mapper[type(expr)]
        except KeyError:
            pass

        # NOTE: in the future we might want to track down other kinds
        # of operations here (e.g., memcpy). However, we don't care for
        # now, since they would remain unexploited inside the compiler

        return None


OpInc = Operation('+')
OpMax = Operation('max')
OpMin = Operation('min')


class LoweredEq(IREq):

    """
    LoweredEq(devito.Eq)
    LoweredEq(devito.LoweredEq, **kwargs)
    LoweredEq(lhs, rhs, **kwargs)

    A SymPy equation enriched with metadata such as an IterationSpace.

    When created as `LoweredEq(devito.Eq)`, the iteration space is automatically
    derived from analysis of `expr`.

    When created as `LoweredEq(devito.LoweredEq, **kwargs)`, the keyword
    arguments can be anything that appears in `LoweredEq.__rkwargs__`
    (e.g., ispace).

    When created as `LoweredEq(lhs, rhs, **kwargs)`, *all* keywords in
    `LoweredEq.__rkwargs__` must appear in `kwargs`.
    """

    __rkwargs__ = IREq.__rkwargs__ + ('reads', 'writes')

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], LoweredEq):
            # origin: LoweredEq(devito.LoweredEq, **kwargs)
            input_expr = args[0]
            expr = sympy.Eq.__new__(cls, *input_expr.args, evaluate=False)
            for i in cls.__rkwargs__:
                setattr(expr, '_%s' % i, kwargs.get(i) or getattr(input_expr, i))
            return expr
        elif len(args) == 1 and isinstance(args[0], Eq):
            # origin: LoweredEq(devito.Eq)
            input_expr = expr = args[0]
        elif len(args) == 2:
            expr = sympy.Eq.__new__(cls, *args, evaluate=False)
            for i in cls.__rkwargs__:
                setattr(expr, '_%s' % i, kwargs.pop(i))
            return expr
        else:
            raise ValueError("Cannot construct LoweredEq from args=%s "
                             "and kwargs=%s" % (str(args), str(kwargs)))

        # Well-defined dimension ordering
        ordering = dimension_sort(expr)

        # Analyze the expression
        accesses = detect_accesses(expr)
        dimensions = Stencil.union(*accesses.values())

        # Separate out the SubIterators from the main iteration Dimensions, that
        # is those which define an actual iteration space
        iterators = {}
        for d in dimensions:
            if d.is_SubIterator:
                iterators.setdefault(d.root, set()).add(d)
            elif d.is_Conditional:
                # Use `parent`, and `root`, because a ConditionalDimension may
                # have a SubDimension as parent
                iterators.setdefault(d.parent, set())
            elif not d.is_Stencil:
                iterators.setdefault(d, set())

        # Construct the IterationSpace
        intervals = IntervalGroup([Interval(d, 0, 0) for d in iterators],
                                  relations=ordering.relations)
        ispace = IterationSpace(intervals, iterators)

        # Construct the conditionals and replace the ConditionalDimensions in `expr`
        conditionals = {}
        for d in ordering:
            if not d.is_Conditional:
                continue
            if d.condition is None:
                conditionals[d] = GuardFactor(d)
            else:
                conditionals[d] = diff2sympy(lower_exprs(d.condition))
            if d.factor is not None:
                expr = uxreplace(expr, {d: IntDiv(d.index, d.factor)})
        conditionals = frozendict(conditionals)

        # Lower all Differentiable operations into SymPy operations
        rhs = diff2sympy(expr.rhs)

        # Finally create the LoweredEq with all metadata attached
        expr = super(LoweredEq, cls).__new__(cls, expr.lhs, rhs, evaluate=False)

        expr._ispace = ispace
        expr._conditionals = conditionals
        expr._reads, expr._writes = detect_io(expr)
        expr._implicit_dims = input_expr.implicit_dims
        expr._operation = Operation.detect(input_expr)

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
        return self._rebuild(*args, evaluate=False)


class ClusterizedEq(IREq):

    """
    ClusterizedEq(devito.IREq, **kwargs)
    ClusterizedEq(lhs, rhs, **kwargs)

    A SymPy equation enriched with metadata such as an IterationSpace.

    There are two main differences between a LoweredEq and a
    ClusterizedEq:

    * To construct a ClusterizedEq, the IterationSpace must be provided
      by the caller, whie in a LoweredEq the IterationSpace is derived
      by analysis of the input.
    * A ClusterizedEq is "frozen", meaning that any call to e.g. `xreplace`
      will not trigger re-evaluation (e.g., mathematical simplification)
      of the expression.

    These two properties make a ClusterizedEq suitable for use in a Cluster.
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            # origin: ClusterizedEq(expr, **kwargs)
            input_expr = args[0]
            expr = sympy.Eq.__new__(cls, *input_expr.args, evaluate=False)
            if isinstance(input_expr, IREq):
                for i in cls.__rkwargs__:
                    try:
                        v = kwargs[i]
                    except KeyError:
                        v = getattr(input_expr, i, None)
                    setattr(expr, '_%s' % i, v)
            else:
                expr._ispace = kwargs['ispace']
                expr._conditionals = kwargs.get('conditionals', frozendict())
                expr._implicit_dims = input_expr.implicit_dims
                expr._operation = Operation.detect(input_expr)
        elif len(args) == 2:
            # origin: ClusterizedEq(lhs, rhs, **kwargs)
            expr = sympy.Eq.__new__(cls, *args, evaluate=False)
            for i in cls.__rkwargs__:
                setattr(expr, '_%s' % i, kwargs.pop(i))
        else:
            raise ValueError("Cannot construct ClusterizedEq from args=%s "
                             "and kwargs=%s" % (str(args), str(kwargs)))
        return expr

    func = IREq._rebuild


class DummyEq(ClusterizedEq):

    """
    DummyEq(expr)
    DummyEq(lhs, rhs)

    A special ClusterizedEq with a void iteration space.
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            input_expr = args[0]
            assert isinstance(input_expr, Eq)
            obj = LoweredEq(input_expr)
        elif len(args) == 2:
            obj = LoweredEq(Eq(*args, evaluate=False))
        else:
            raise ValueError("Cannot construct DummyEq from args=%s" % str(args))
        return ClusterizedEq.__new__(cls, obj, ispace=obj.ispace)

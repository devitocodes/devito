from contextlib import suppress
from functools import cached_property, singledispatch

import numpy as np
import sympy

from devito.finite_differences.differentiable import diff2sympy
from devito.ir.equations.algorithms import dimension_sort, lower_exprs
from devito.ir.support import (
    GuardFactor, Interval, IntervalGroup, IterationSpace, Stencil, detect_accesses
)
from devito.symbolics import IntDiv, limits_mapper, retrieve_accesses, uxreplace
from devito.tools import (
    Pickable, Tag, as_hashable, filter_sorted, frozendict, reuse_if_unchanged
)
from devito.types import (
    Eq, Inc, IncrInterpolation, Injection, InjectionMixin, Interpolation,
    InterpolationMixin, ReduceMax, ReduceMin, ReduceMinMax, SparseEq, SparseOpMixin,
    relational_min
)

__all__ = [
    'ClusterizedEq',
    'ClusterizedIncrInterpolation',
    'ClusterizedInjection',
    'ClusterizedInterpolation',
    'ClusterizedSparseEq',
    'DummyEq',
    'LoweredEq',
    'LoweredIncrInterpolation',
    'LoweredInjection',
    'LoweredInterpolation',
    'LoweredSparseEq',
    'OpInc',
    'OpMax',
    'OpMin',
    'OpMinMax',
    'clusterize_eq',
    'identity_mapper',
    'lower_eq',
]


class IREq(sympy.Eq, Pickable):

    __rargs__ = ('lhs', 'rhs')
    __rkwargs__ = ('ispace', 'conditionals', 'implicit_dims', 'operation')

    is_SparseOperation = False

    def _hashable_content(self):
        return (super()._hashable_content() +
                tuple(as_hashable(getattr(self, i)) for i in self.__rkwargs__))

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
        return self.operation in (OpInc, OpMin, OpMax, OpMinMax)

    @property
    def is_Increment(self):
        return self.operation is OpInc

    @cached_property
    def writes(self):
        from devito.symbolics.queries import q_routine

        terminals = set(retrieve_accesses(self.lhs))
        if q_routine(self.rhs):
            with suppress(AttributeError):
                # Everything except: foreign routines, such as `cos` or `sin` etc.
                terminals.update(self.rhs.writes)

        return tuple(terminals)

    @cached_property
    def reads_explicit(self):
        terminals = set(retrieve_accesses(self.rhs, deep=True))
        with suppress(AttributeError):
            terminals.update(retrieve_accesses(self.lhs.indices))

        return tuple(terminals)

    @cached_property
    def reads_conditional(self):
        accesses = []
        for v in self.conditionals.values():
            accesses.extend(retrieve_accesses(v))

        return tuple(accesses)

    @cached_property
    def reads(self):
        return tuple(set(self.reads_explicit) | set(self.reads_conditional))

    @cached_property
    def _read_functions(self):
        found = []
        for i in self.reads:
            with suppress(AttributeError):
                i = i.function
            found.append(i)
        return tuple(filter_sorted(found))

    @cached_property
    def _write_functions(self):
        found = []
        for i in self.writes:
            with suppress(AttributeError):
                i = i.function
            found.append(i)
        return tuple(filter_sorted(found))

    @cached_property
    def read_functions(self):
        return tuple(i for i in self._read_functions if i.is_Input)

    @cached_property
    def write_functions(self):
        return tuple(i for i in self._write_functions if i.is_Input)

    @cached_property
    def read_functions_relaxed(self):
        return tuple(i for i in self._read_functions
                     if i.is_Input or i.is_AbstractFunction)

    @cached_property
    def write_functions_relaxed(self):
        return tuple(i for i in self._write_functions
                     if i.is_Input or i.is_AbstractFunction)

    def apply(self, func):
        """
        Apply a callable to `self` and each expr-like attribute carried by `self`,
        thus triggering a reconstruction.
        """
        args = [func(self.lhs), func(self.rhs)]
        kwargs = dict(self.state)

        conditionals = {k: func(v) for k, v in self.conditionals.items()}
        kwargs['conditionals'] = frozendict(conditionals)

        return self.func(*args, **kwargs)

    def __repr__(self):
        if not self.is_Reduction:
            return super().__repr__()
        elif self.operation is OpInc:
            return f'Inc({self.lhs}, {self.rhs})'
        else:
            return f'Eq({self.lhs}, {self.operation}({self.rhs}))'

    __str__ = __repr__

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class Operation(Tag):

    """
    Special operation performed by an Eq.
    """

    @classmethod
    def detect(cls, expr):
        reduction_mapper = (
            (ReduceMinMax, OpMinMax),
            (ReduceMin, OpMin),
            (ReduceMax, OpMax),
            (Inc, OpInc),
            # An ``Interpolation`` looks like a plain ``Eq`` -- ``sf[p_*] =
            # expr[rp_*]`` -- but the cluster scheduler iterates the rhs
            # over the radius dims, so values are implicitly summed across
            # ``rp_*``. Tagging it as ``OpInc`` makes the dependence
            # analysis treat ``rp_*`` as reduction dims
            # (``parallel_if_atomic``), which matches the lowered code
            # (``sumrec += ... ; sf[p_*] = sumrec``) and stops the
            # blocking heuristic from turning the tiny radius loops into
            # thread blocks. The actual write-back flavour at ``sf[p_*]``
            # is keyed off the Eq's class (``is_increment_writeback``) in
            # ``lower_sparse_ops`` so this tag doesn't accidentally turn
            # ``Interpolation`` assignments into increments.
            (InterpolationMixin, OpInc),
        )
        for kls, op in reduction_mapper:
            if isinstance(expr, kls):
                return op

        # NOTE: in the future we might want to track down other kinds
        # of operations here (e.g., memcpy). However, we don't care for
        # now, since they would remain unexploited inside the compiler

        return None


OpInc = Operation('+')
OpMax = Operation('max')
OpMin = Operation('min')
OpMinMax = Operation('minmax')


identity_mapper = {
    np.int32: {OpInc: sympy.S.Zero,
               OpMax: limits_mapper[np.int32].min,
               OpMin: limits_mapper[np.int32].max},
    np.int64: {OpInc: sympy.S.Zero,
               OpMax: limits_mapper[np.int64].min,
               OpMin: limits_mapper[np.int64].max},
    np.float32: {OpInc: sympy.S.Zero,
                 OpMax: limits_mapper[np.float32].min,
                 OpMin: limits_mapper[np.float32].max},
    np.float64: {OpInc: sympy.S.Zero,
                 OpMax: limits_mapper[np.float64].min,
                 OpMin: limits_mapper[np.float64].max},
}


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

    __rkwargs__ = IREq.__rkwargs__

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], LoweredEq):
            # origin: LoweredEq(devito.LoweredEq, **kwargs)
            input_expr = args[0]
            expr = sympy.Eq.__new__(cls, *input_expr.args, evaluate=False)
            for i in cls.__rkwargs__:
                setattr(expr, f'_{i}', kwargs.get(i) or getattr(input_expr, i))
            return expr
        elif len(args) == 1 and isinstance(args[0], Eq):
            # origin: LoweredEq(devito.Eq)
            input_expr = expr = args[0]
        elif len(args) == 2:
            expr = sympy.Eq.__new__(cls, *args, evaluate=False)
            for i in cls.__rkwargs__:
                setattr(expr, f'_{i}', kwargs.pop(i))
            return expr
        else:
            raise ValueError(f"Cannot construct LoweredEq from args={str(args)} "
                             f"and kwargs={str(kwargs)}")

        # Well-defined dimension ordering
        ordering = dimension_sort(expr)

        # Analyze the expression
        accesses = detect_accesses(expr)
        dimensions = Stencil.union(*accesses.values())

        # Separate out the SubIterators from the main iteration
        # Dimensions, that is those which define an actual
        # iteration space
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
        intervals = IntervalGroup([Interval(d) for d in iterators],
                                  relations=ordering.relations, mode='partial')
        ispace = IterationSpace(intervals, iterators)

        # Construct the conditionals and replace the ConditionalDimensions in `expr`
        conditionals = {}
        for d in ordering:
            if not d.is_Conditional:
                continue
            if d.condition is None:
                conditionals[d] = GuardFactor(d)
            else:
                cond = diff2sympy(lower_exprs(d.condition))
                if d._factor is not None:
                    cond = d.relation(cond, GuardFactor(d))
                conditionals[d] = cond
            # Replace dimension with index
            index = d.index
            if d.condition is not None and d in expr.free_symbols:
                index = index - relational_min(d.condition, d.parent)
            expr = uxreplace(expr, {d: IntDiv(index, d.symbolic_factor)})

        conditionals = frozendict(conditionals)

        # Lower all Differentiable operations into SymPy operations
        rhs = diff2sympy(expr.rhs)

        # Finally create the LoweredEq with all metadata attached
        expr = super().__new__(cls, expr.lhs, rhs, evaluate=False)

        expr._ispace = ispace
        expr._conditionals = conditionals
        expr._implicit_dims = input_expr.implicit_dims
        expr._operation = Operation.detect(input_expr)

        return expr

    def xreplace(self, rules):
        return LoweredEq(self.lhs.xreplace(rules), self.rhs.xreplace(rules), **self.state)

    def func(self, *args):
        return self._rebuild(*args, evaluate=False)


class LoweredSparseEq(SparseOpMixin, LoweredEq):

    """
    The IR counterpart of ``SparseEq``: a regular ``LoweredEq`` that
    also carries the ``interpolator`` metadata used by the IET pass
    ``lower_sparse_ops`` to wrap the resulting ``p_*, rp_*`` iteration
    nest in an ElementalFunction. Subclassed by
    ``LoweredInterpolation`` / ``LoweredIncrInterpolation`` /
    ``LoweredInjection`` for the per-operation polymorphic behaviour.
    """

    __rkwargs__ = LoweredEq.__rkwargs__ + ('interpolator',)


class LoweredInterpolation(InterpolationMixin, LoweredSparseEq):
    """IR counterpart of ``Interpolation``."""
    # ``sf[p_*] = ...``: the write-back at the sparse position is an
    # assignment. The Eq is still tagged as a reduction
    # (``OpInc``/``is_Reduction``) because the rhs is summed over the
    # radius dims; only the *outer* write-back to ``sf[p_*]`` is plain.
    is_increment_writeback = False


class LoweredIncrInterpolation(InterpolationMixin, LoweredSparseEq):
    """IR counterpart of ``IncrInterpolation``."""
    # ``sf[p_*] += ...``: the user asked for ``interpolate(..., increment=True)``.
    is_increment_writeback = True


class LoweredInjection(InjectionMixin, LoweredSparseEq):
    """IR counterpart of ``Injection``."""
    pass


# Map user-level sparse-op classes to their IR-level counterparts.
_lowered_sparse_cls = {
    Interpolation: LoweredInterpolation,
    IncrInterpolation: LoweredIncrInterpolation,
    Injection: LoweredInjection,
}


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

    @reuse_if_unchanged('__rkwargs__')
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
                    setattr(expr, f'_{i}', v)
            else:
                expr._ispace = kwargs['ispace']
                expr._conditionals = kwargs.get('conditionals', {})
                expr._implicit_dims = input_expr.implicit_dims
                expr._operation = Operation.detect(input_expr)
        elif len(args) == 2:
            # origin: ClusterizedEq(lhs, rhs, **kwargs)
            expr = sympy.Eq.__new__(cls, *args, evaluate=False)
            for i in cls.__rkwargs__:
                setattr(expr, f'_{i}', kwargs.pop(i))
        else:
            raise ValueError(f"Cannot construct ClusterizedEq from args={str(args)} "
                             f"and kwargs={str(kwargs)}")

        # Immutability (and thus hashability, etc)
        expr._conditionals = frozendict(expr._conditionals)

        return expr

    func = IREq._rebuild


class ClusterizedSparseEq(SparseOpMixin, ClusterizedEq):

    """
    Frozen counterpart of ``LoweredSparseEq``: the same regular
    ``ClusterizedEq`` augmented with ``interpolator`` so the IET pass
    ``lower_sparse_ops`` can identify and rewrite the sparse op's
    iteration nest. Subclassed by ``ClusterizedInterpolation`` /
    ``ClusterizedIncrInterpolation`` / ``ClusterizedInjection``.
    """

    __rkwargs__ = ClusterizedEq.__rkwargs__ + ('interpolator',)


class ClusterizedInterpolation(InterpolationMixin, ClusterizedSparseEq):
    """Frozen counterpart of ``LoweredInterpolation``."""
    is_increment_writeback = False


class ClusterizedIncrInterpolation(InterpolationMixin, ClusterizedSparseEq):
    """Frozen counterpart of ``LoweredIncrInterpolation``."""
    is_increment_writeback = True


class ClusterizedInjection(InjectionMixin, ClusterizedSparseEq):
    """Frozen counterpart of ``LoweredInjection``."""
    pass


_clusterized_sparse_cls = {
    LoweredInterpolation: ClusterizedInterpolation,
    LoweredIncrInterpolation: ClusterizedIncrInterpolation,
    LoweredInjection: ClusterizedInjection,
}


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
            raise ValueError(f"Cannot construct DummyEq from args={str(args)}")
        return ClusterizedEq.__new__(cls, obj, ispace=obj.ispace)


@singledispatch
def lower_eq(eq):
    """
    Promote a user-level ``Eq`` to its ``LoweredEq`` counterpart, ready
    for the cluster pipeline. The dispatch matches the dynamic type of
    ``eq``; ``SparseEq`` and friends get their own branch.
    """
    return LoweredEq(eq)


@lower_eq.register(SparseEq)
def _(eq):
    # Augment ``implicit_dims`` with the SparseFunction's own iteration
    # Dimensions (e.g. ``p_sf`` and any extra SparseFunction dims) so
    # the cluster scheduler sees them. Grid Dimensions reached through
    # the rhs Function are deliberately *not* added: SubDomain-derived
    # SubDimensions would otherwise spuriously appear in the
    # IterationSpace, and grid Dimensions are already discovered via
    # the radius ConditionalDimensions in the rhs.
    interp = eq.interpolator
    sf_dims = tuple(interp.sfunction.dimensions)
    user = tuple(eq.implicit_dims or ())
    if interp.sfunction._sparse_position == -1:
        augmented = sf_dims + user
    else:
        augmented = user + sf_dims

    if augmented != tuple(eq.implicit_dims or ()):
        eq = eq.func(eq.lhs, eq.rhs, interpolator=interp,
                     implicit_dims=augmented)

    lowered_cls = _lowered_sparse_cls[type(eq)]
    obj = lowered_cls(eq)
    obj._interpolator = interp
    return obj


@singledispatch
def clusterize_eq(eq, **kwargs):
    """
    Freeze a ``LoweredEq`` into its ``ClusterizedEq`` counterpart,
    suitable for use in a ``Cluster``. Subclasses with extra payload
    (e.g. ``LoweredSparseEq``) dispatch to their frozen counterpart.
    """
    return ClusterizedEq(eq, **kwargs)


@clusterize_eq.register(LoweredSparseEq)
def _(eq, **kwargs):
    return _clusterized_sparse_cls[type(eq)](eq, **kwargs)


@clusterize_eq.register(ClusterizedSparseEq)
def _(eq, **kwargs):
    # ``eq`` is already clusterized; rebuild via its own class to preserve
    # the per-operation polymorphic behaviour.
    return type(eq)(eq, **kwargs)

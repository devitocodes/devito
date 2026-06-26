"""
Guards are symbolic relationals that can be used in different layers
of the compiler to express the conditions under which a certain object
(e.g., Eq, Cluster, ...) should be evaluated at runtime.
"""

from collections import Counter, defaultdict
from functools import singledispatch
from operator import ge, gt, le, lt

import numpy as np
from sympy import And, Expr, Ge, Gt, Le, Lt, Mul, true
from sympy.logic.boolalg import BooleanFunction

from devito.ir.support.space import Forward, IterationDirection
from devito.symbolics import CondEq, CondNe, IntDiv, search
from devito.symbolics.manipulation import _uxreplace_handle, _uxreplace_registry
from devito.tools import Pickable, as_tuple, frozendict, split
from devito.types import Dimension, LocalObject

__all__ = [
    'BaseGuardBound',
    'BaseGuardBoundNext',
    'GuardBound',
    'GuardBoundNext',
    'GuardCaseSwitch',
    'GuardExpr',
    'GuardFactor',
    'GuardOverflow',
    'GuardSwitch',
    'Guards',
]


class AbstractGuard:
    pass


class Guard(AbstractGuard):

    @property
    def _args_rebuild(self):
        return self.args

    @property
    def canonical(self):
        return self

    @property
    def negated(self):
        return negations[self.__class__](*self._args_rebuild, evaluate=False)


# *** GuardFactor


class GuardFactor(Guard, CondEq, Pickable):

    """
    A guard for factor-based ConditionalDimensions.

    Given the ConditionalDimension `d` with factor `k`, create the
    symbolic relational `d.parent % k == 0`.
    """

    __rargs__ = ('d',)

    def __new__(cls, d, **kwargs):
        assert d.is_Conditional

        obj = super().__new__(cls, d.parent % d.symbolic_factor, 0)
        obj.d = d

        return obj

    @property
    def _args_rebuild(self):
        return (self.d,)


class GuardFactorEq(GuardFactor, CondEq):
    pass


class GuardFactorNe(GuardFactor, CondNe):
    pass


GuardFactor = GuardFactorEq


# *** GuardBound


class BaseGuardBound(Guard):

    """
    A guard to avoid out-of-bounds iteration.

    Given two iteration points `p0` and `p1`, create the symbolic
    relational `p0 <= p1`.
    """

    def __new__(cls, p0, p1, **kwargs):
        try:
            if cls.__base__._eval_relation(p0, p1) is true:
                return None
        except TypeError:
            pass
        return super().__new__(cls, p0, p1, evaluate=False)


class GuardBoundLe(BaseGuardBound, Le):
    pass


class GuardBoundGt(BaseGuardBound, Gt):
    pass


GuardBound = GuardBoundLe


# *** GuardBoundNext


class BaseGuardBoundNext(Guard, Pickable):

    """
    A guard to avoid out-of-bounds iteration.

    Given a Dimension `d` and an IterationDirection `direction`, create a
    symbolic relation that at runtime evaluates to true if

        * `next(d) <= d.root.symbolic_max`, with `direction=Forward`
        * `d.root.symbolic_min <= next(d)`, with `direction=Backward`

    where `next(d)` represents the next iteration along `d` for the
    given `direction`.
    """

    __rargs__ = ('d', 'index', 'direction')
    __rkwargs__ = ('d_min', 'd_max')

    def __new__(cls, d, index, direction,
                d_min=None, d_max=None, **kwargs):
        assert isinstance(d, Dimension)
        assert isinstance(direction, IterationDirection)

        # Always take the next index in the iteration direction
        next_index = eval_next_index(index, d, direction)

        # The direction might be forward but accessing c - d
        # making the access backward w.r.t
        # Update direction according to access direction for valid guard
        if index.has(-d):
            direction = -direction

        if direction == Forward:
            p0 = next_index
            p1 = d_max or d.root.symbolic_max
        else:
            p0 = d_min if d_min is not None else d.root.symbolic_min
            p1 = next_index

        try:
            if cls.__base__._eval_relation(p0, p1) is true:
                return None
        except TypeError:
            pass

        obj = super().__new__(cls, p0, p1, evaluate=False)

        obj.d = d
        obj.direction = direction
        obj.index = index
        obj.d_min = d_min
        obj.d_max = d_max

        return obj

    @property
    def _args_rebuild(self):
        return (self.d, self.index, self.direction)


class GuardBoundNextLe(BaseGuardBoundNext, Le):
    pass


class GuardBoundNextGt(BaseGuardBoundNext, Gt):
    pass


GuardBoundNext = GuardBoundNextLe


class BaseGuardOverflow(Guard):

    """
    A guard for buffer overflow.
    """

    pass


class GuardOverflowGe(BaseGuardOverflow, Ge):
    pass


class GuardOverflowLt(BaseGuardOverflow, Lt):
    pass


GuardOverflow = GuardOverflowGe


negations = {
    GuardFactorEq: GuardFactorNe,
    GuardFactorNe: GuardFactorEq,
    GuardBoundLe: GuardBoundGt,
    GuardBoundGt: GuardBoundLe,
    GuardBoundNextLe: GuardBoundNextGt,
    GuardBoundNextGt: GuardBoundNextLe,
    GuardOverflowGe: GuardOverflowLt,
    GuardOverflowLt: GuardOverflowGe
}


class GuardSwitch(AbstractGuard, Expr):

    """
    A switch guard (akin to C's switch-case) that can be used to select
    between multiple cases at runtime.
    """

    def __new__(cls, arg, **kwargs):
        return Expr.__new__(cls, arg)

    @property
    def arg(self):
        return self.args[0]


class GuardCaseSwitch(GuardSwitch):

    """
    A case within a GuardSwitch.
    """

    def __new__(cls, arg, case, **kwargs):
        return Expr.__new__(cls, arg, case)

    @property
    def case(self):
        return self.args[1]


class Guards(frozendict):

    """
    A mapper {Dimension -> guard}.

    The mapper is immutable; operations such as andg, get, ... cause the
    construction of a new object, leaving self intact.
    """

    def get(self, d, v=true):
        return super().get(d, v)

    def _reuse_if_untouched(self, mapper):
        return self if mapper == self else Guards(mapper)

    def andg(self, d, guard):
        m = dict(self)

        if guard == true:
            return self

        try:
            m[d] = simplify_and(m[d], guard)
        except KeyError:
            m[d] = guard

        return self._reuse_if_untouched(m)

    def xandg(self, d, guard):
        m = dict(self)

        if guard == true:
            return self

        try:
            m[d] = And(m[d], guard)
        except KeyError:
            m[d] = guard

        return self._reuse_if_untouched(m)

    def pairwise_or(self, d, *guards):
        m = dict(self)
        guards = list(guards)

        if d in m:
            guards.append(m[d])

        g = pairwise_or(*guards)
        if g is true:
            m.pop(d, None)
        else:
            m[d] = g

        return self._reuse_if_untouched(m)

    def impose(self, d, guard):
        m = dict(self)

        if guard == true:
            return self

        m[d] = guard

        return self._reuse_if_untouched(m)

    def popany(self, dims):
        m = dict(self)

        for d in as_tuple(dims):
            m.pop(d, None)

        return self._reuse_if_untouched(m)

    def filter(self, key):
        m = {d: v for d, v in self.items() if key(d)}

        return self._reuse_if_untouched(m)

    def as_map(self, d, cls):
        if cls not in (Le, Lt, Ge, Gt):
            raise ValueError(f"Unsupported class {cls}")

        return dict(i.args for i in search(self.get(d), cls))


class GuardExpr(LocalObject, BooleanFunction):

    """
    A boolean symbol that can be used as a guard. As such, it can be chained
    with other relations using the standard boolean operators (&, |, ...).

    Being a LocalObject, a GuardExpr may carry an `initvalue`, which is
    the value that the guard assumes at the beginning of the scope where
    it is defined.
    """

    dtype = np.bool_

    def __init__(self, name, liveness='eager', **kwargs):
        super().__init__(name, liveness=liveness, **kwargs)

    @singledispatch
    def _handle_boolean(obj, mapper):
        raise NotImplementedError(f"Cannot handle boolean of type {type(obj)}")

    @_handle_boolean.register(And)
    def _(obj, mapper):
        for a in obj.args:
            GuardExpr._handle_boolean(a, mapper)

    @_handle_boolean.register(Le)
    @_handle_boolean.register(Ge)
    @_handle_boolean.register(Lt)
    @_handle_boolean.register(Gt)
    def _(obj, mapper):
        d, v = obj.args
        k = obj.__class__
        mapper.setdefault(k, {})[d] = v

    @property
    def as_mapper(self):
        mapper = {}
        GuardExpr._handle_boolean(self.initvalue, mapper)
        return frozendict(mapper)

    def sort_key(self, order=None):
        # Use the overarching LocalObject name for arguments ordering
        class_key, args, exp, coeff = super().sort_key(order=order)
        args = (len(args[1]) + 1, (self.name,) + args[1])
        return class_key, args, exp, coeff


# *** Utils

op_mapper = {
    Le: le,
    Lt: lt,
    Ge: ge,
    Gt: gt
}


def simplify_and(relation, v):
    """
    Given `x = And(*relation.args, v)`, return `relation` if `x ≡ relation`,
    `x` otherwise.

    SymPy doesn't have a builtin function to simplify boolean inequalities; here,
    we run a set of simple checks to at least discard the most obvious (and thus
    annoying to see in the generated code) redundancies.
    """
    if isinstance(relation, And):
        candidates, other = split(list(relation.args), lambda a: type(a) is type(v))
    elif type(relation) is type(v):
        candidates, other = [relation], []
    else:
        candidates, other = [], [relation, v]

    covered = False
    new_args = []
    for a in candidates:
        if isinstance(v, GuardExpr) and isinstance(a, GuardExpr):
            # Attempt optimizing guards in GuardExpr form
            covered = True

            m0 = v.as_mapper
            m1 = a.as_mapper

            for cls, op in op_mapper.items():
                if cls in m0 and cls in m1:
                    try:
                        if set(m0[cls]) != set(m1[cls]):
                            new_args.extend([a, v])
                        elif all(op(m0[cls][d], m1[cls][d]) for d in m0[cls]):
                            new_args.append(v)
                        elif all(op(m1[cls][d], m0[cls][d]) for d in m1[cls]):
                            new_args.append(a)
                        else:
                            new_args.extend([a, v])
                    except TypeError:
                        # E.g., `cls = Le`, then `z <= 2` and `z <= z_M + 1`
                        new_args.extend([a, v])

                elif cls in m0:
                    new_args.append(v)

                elif cls in m1:
                    new_args.append(a)

        elif a.lhs is not v.lhs:
            new_args.append(a)

        else:
            # Attempt optimizing guards in relational form
            covered = True

            try:
                if type(a) in (Gt, Ge) \
                        and v.rhs > a.rhs or type(a) in (Lt, Le) \
                        and v.rhs < a.rhs:
                    new_args.append(v)
                else:
                    new_args.append(a)
            except TypeError:
                # E.g., `v.rhs = const + z_M` and `a.rhs = z_M`, so the inequalities
                # above are not evaluable to True/False
                new_args.append(a)

    if not covered:
        new_args.append(v)

    return And(*(new_args + other))


def pairwise_or(*guards):
    """
    Given a series of guards, create a new guard that combines them by taking
    the OR of each subset of homogeneous components. Here, "homogeneous" means
    that they apply to the same variable with the same operator (e.g., given
    `y >= 2`, `y >= 3` is homogeneous, while `z >= 3` and `y <= 4` are not).

    Examples
    --------
    Given:

        g0 = {flag0 and z >= 2 and z <= 10 and y >= 3}
        g1 = {z >= 4 and z <= 8}
        g2 = {y >= 2 and y <= 5}

    Where `flag0` is of type GuardExpr, then:

    Return:

        {z >= 2 and z <= 10 and y >= 2}
    """
    errmsg = lambda g: f"Cannot handle guard component of type {type(g)}"

    flags = Counter()
    mapper = defaultdict(list)

    # Analysis
    for guard in guards:
        if guard is true:
            return true
        elif guard is None:
            continue
        elif isinstance(guard, And):
            components = guard.args
        elif isinstance(guard, GuardExpr) or guard.is_Relational:
            components = [guard]
        else:
            raise NotImplementedError(errmsg(guard))

        for g in components:
            if isinstance(g, GuardExpr):
                flags[g] += 1
            elif g.is_Relational and g.lhs.is_Symbol and g.rhs.is_Number:
                mapper[(g.lhs, type(g))].append(g.rhs)
            else:
                raise NotImplementedError(errmsg(g))

    # Synthesis
    guard = true
    for (s, op), v in mapper.items():
        if len(v) < len(guards):
            # Not all guards contributed to this component; cannot simplify
            pass
        elif op in (Ge, Gt):
            guard = And(guard, op(s, min(v)))
        else:
            guard = And(guard, op(s, max(v)))

    for flag, v in flags.items():
        if v == len(guards):
            guard = And(guard, flag)
        elif flag.initvalue.free_symbols & guard.free_symbols:
            # We still lack the logic to properly handle this case
            raise NotImplementedError(errmsg(flag))
        else:
            # Safe to ignore
            pass

    return guard


_uxreplace_registry.register(BaseGuardBoundNext)


@_uxreplace_handle.register(BaseGuardBoundNext)
def _(expr, args, kwargs):
    return expr.func(expr.d, expr.index, expr.direction, **kwargs)


@singledispatch
def eval_next_index(expr, dim, dir):
    """
    Evaluate `expr` at the next iteration point along `dim` in the given
    `dir`-ection. The "next" point is obtained by substituting `dim` with
    `dim + 1` for `Forward` and `dim - 1` for `Backward`.

    For `IntDiv` expressions encoding subsampling (`dim.root // factor`),
    the result is rounded to the next valid coarse-grained slot.
    """
    if dir == Forward:
        return expr._subs(dim, dim + 1)
    else:
        return expr._subs(dim, dim - 1)


@eval_next_index.register(Expr)
def _(expr, dim, dir):
    if not expr.args:
        if dir == Forward:
            return expr._subs(dim, dim + 1)
        else:
            return expr._subs(dim, dim - 1)
    return expr.func(*[eval_next_index(a, dim, dir) for a in expr.args])


@eval_next_index.register(IntDiv)
def _(expr, dim, dir):
    v = dim.symbolic_factor
    p0 = dim.root
    if dir == Forward:
        return Mul((((p0 + 1) + v - 1) / v), v, evaluate=False)
    else:
        return (p0 - 1) - abs(p0 - 1) % v

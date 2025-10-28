"""
Guards are symbolic relationals that can be used in different layers
of the compiler to express the conditions under which a certain object
(e.g., Eq, Cluster, ...) should be evaluated at runtime.
"""

from collections import Counter, defaultdict
from operator import ge, gt, le, lt

from functools import singledispatch
from sympy import And, Ge, Gt, Le, Lt, Mul, true
from sympy.logic.boolalg import BooleanFunction
import numpy as np

from devito.ir.support.space import Forward, IterationDirection
from devito.symbolics import CondEq, CondNe
from devito.tools import Pickable, as_tuple, frozendict, split
from devito.types import Dimension, LocalObject

__all__ = ['GuardFactor', 'GuardBound', 'GuardBoundNext', 'BaseGuardBound',
           'BaseGuardBoundNext', 'GuardOverflow', 'Guards', 'GuardExpr']


class Guard:

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

    __rargs__ = ('d', 'direction')

    def __new__(cls, d, direction, **kwargs):
        assert isinstance(d, Dimension)
        assert isinstance(direction, IterationDirection)

        if direction == Forward:
            p0 = d.root
            p1 = d.root.symbolic_max

            if d.is_Conditional:
                v = d.symbolic_factor
                # Round `p0 + 1` up to the nearest multiple of `v`
                p0 = Mul((((p0 + 1) + v - 1) / v), v, evaluate=False)
            else:
                p0 = p0 + 1

        else:
            p0 = d.root.symbolic_min
            p1 = d.root

            if d.is_Conditional:
                v = d.symbolic_factor
                # Round `p1 - 1` down to the nearest sub-multiple of `v`
                # NOTE: we use ABS to make sure we handle negative values properly.
                # Once `p1 - 1` is negative (e.g. `iteration=time - 1` and `time=0`),
                # as long as we get a negative number, rather than 0 and even if it's
                # not `-v`, we're good
                p1 = (p1 - 1) - abs(p1 - 1) % v
            else:
                p1 = p1 - 1

        try:
            if cls.__base__._eval_relation(p0, p1) is true:
                return None
        except TypeError:
            pass

        obj = super().__new__(cls, p0, p1, evaluate=False)

        obj.d = d
        obj.direction = direction

        return obj

    @property
    def _args_rebuild(self):
        return (self.d, self.direction)


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


class Guards(frozendict):

    """
    A mapper {Dimension -> guard}.

    The mapper is immutable; operations such as andg, get, ... cause the
    construction of a new object, leaving self intact.
    """

    def get(self, d, v=true):
        return super().get(d, v)

    def andg(self, d, guard):
        m = dict(self)

        if guard == true:
            return Guards(m)

        try:
            m[d] = simplify_and(m[d], guard)
        except KeyError:
            m[d] = guard

        return Guards(m)

    def xandg(self, d, guard):
        m = dict(self)

        if guard == true:
            return Guards(m)

        try:
            m[d] = And(m[d], guard)
        except KeyError:
            m[d] = guard

        return Guards(m)

    def pairwise_or(self, d, *guards):
        m = dict(self)

        if d in m:
            guards.append(m[d])

        g = pairwise_or(*guards)
        if g is true:
            m.pop(d, None)
        else:
            m[d] = g

        return Guards(m)

    def impose(self, d, guard):
        m = dict(self)

        if guard == true:
            return Guards(m)

        m[d] = guard

        return Guards(m)

    def popany(self, dims):
        m = dict(self)

        for d in as_tuple(dims):
            m.pop(d, None)

        return Guards(m)

    def filter(self, key):
        m = {d: v for d, v in self.items() if key(d)}

        return Guards(m)

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

    dtype = np.bool

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
                if type(a) in (Gt, Ge) and v.rhs > a.rhs:
                    new_args.append(v)
                elif type(a) in (Lt, Le) and v.rhs < a.rhs:
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
        if guard is true or guard is None:
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

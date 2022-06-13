"""
Guards are symbolic relationals that can be used in different layers
of the compiler to express the conditions under which a certain object
(e.g., Eq, Cluster, ...) should be evaluated at runtime.
"""

from sympy import Ge, Gt, Le, Lt, Mul, true
from sympy.core.operations import LatticeOp

from devito.ir.support.space import Forward, IterationDirection
from devito.symbolics import CondEq, CondNe, FLOAT
from devito.types import Dimension

__all__ = ['GuardFactor', 'GuardBound', 'GuardBoundNext', 'BaseGuardBound',
           'BaseGuardBoundNext', 'GuardOverflow', 'transform_guard']


class Guard(object):

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


class GuardFactor(Guard, CondEq):

    """
    A guard for factor-based ConditionalDimensions.

    Given the ConditionalDimension `d` with factor `k`, create the
    symbolic relational `d.parent % k == 0`.
    """

    def __new__(cls, d, **kwargs):
        assert d.is_Conditional

        obj = super().__new__(cls, d.parent % d.factor, 0)
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


class BaseGuardBoundNext(Guard):

    """
    A guard to avoid out-of-bounds iteration.

    Given a Dimension `d` and an IterationDirection `direction`, create a
    symbolic relation that at runtime evaluates to true if

        * `next(d) <= d.root.symbolic_max`, with `direction=Forward`
        * `d.root.symbolic_min <= next(d)`, with `direction=Backward`

    where `next(d)` represents the next iteration along `d` for the
    given `direction`.
    """

    def __new__(cls, d, direction, **kwargs):
        assert isinstance(d, Dimension)
        assert isinstance(direction, IterationDirection)

        if direction is Forward:
            p0 = d.root
            p1 = d.root.symbolic_max

            if d.is_Conditional:
                v = d.factor
                # Round `p0` up to the nearest multiple of `v`
                p0 = Mul((((p0 + 1) + v - 1) / v), v, evaluate=False)
            else:
                p0 = p0 + 1

        else:
            p0 = d.root.symbolic_min
            p1 = d.root

            if d.is_Conditional:
                v = d.factor
                # Round `p1` down to the nearest sub-multiple of `v`
                # NOTE: we use FLOAT(d.factor) to make sure we don't drop negative
                # values on the floor. E.g., `iteration=time - 1`, `v=2`, then when
                # `time=0` we want the Mul to evaluate to -1, not to 0, which is
                # what C's integer division would give us
                p1 = Mul(((p1 - 1) / FLOAT(v)), v, evaluate=False)
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


def transform_guard(expr, guard_type, callback):
    """
    Transform the components of a guard according to `callback`.
    A component `c` is transformed iff `isinstance(c, guard_type)`.
    """
    if isinstance(expr, guard_type):
        return callback(expr)
    elif isinstance(expr, LatticeOp):
        return expr.func(*[transform_guard(a, guard_type, callback) for a in expr.args])
    else:
        return expr


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

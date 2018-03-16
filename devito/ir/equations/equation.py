from sympy import Eq

from devito.dimension import SubDimension
from devito.equation import DOMAIN, INTERIOR
from devito.ir.support import (IterationSpace, Any, compute_intervals,
                               compute_directions, detect_io)
from devito.symbolics import FrozenExpr, dimension_sort, indexify

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


class LoweredEq(Eq, IREq):

    """
    LoweredEq(expr, subs=None)

    A SymPy equation with an associated iteration space.

    All :class:`Function` objects within ``expr`` get indexified and thus turned
    into objects of type :class:`types.Indexed`.

    The iteration space is an object of type :class:`IterationSpace`. It
    represents the iteration points along each :class:`Dimension` that the
    equation may traverse with the guarantee that no out-of-bounds accesses
    will be performed.
    """

    def __new__(cls, *args, **kwargs):
        # Parse input
        if len(args) == 1:
            input_expr = args[0]
            assert type(input_expr) != LoweredEq
            assert isinstance(input_expr, Eq)
        elif len(args) == 2:
            # Reconstructing from existing Eq. E.g., we end up here after xreplace
            stamp = kwargs.pop('stamp')
            expr = Eq.__new__(cls, *args, evaluate=False)
            assert isinstance(stamp, Eq)
            expr.is_Increment = stamp.is_Increment
            expr.ispace = stamp.ispace
            return expr
        else:
            raise ValueError("Cannot construct LoweredEq from args=%s "
                             "and kwargs=%s" % (str(args), str(kwargs)))

        # Indexification
        expr = indexify(input_expr)

        # Apply caller-provided substitution
        subs = kwargs.get('subs')
        if subs is not None:
            expr = expr.xreplace(subs)

        # Well-defined dimension ordering
        ordering = dimension_sort(expr, key=lambda i: not i.is_Time)

        # Introduce space sub-dimensions if need to
        region = getattr(input_expr, '_region', DOMAIN)
        if region == INTERIOR:
            mapper = {i: SubDimension("%si" % i, i, 1, -1)
                      for i in ordering if i.is_Space}
            expr = expr.xreplace(mapper)
            ordering = [mapper.get(i, i) for i in ordering]

        # Compute iteration space
        intervals, iterators = compute_intervals(expr)
        intervals = sorted(intervals, key=lambda i: ordering.index(i.dim))
        directions, _ = compute_directions(expr, lambda i: Any)
        ispace = IterationSpace([i.negate() for i in intervals], iterators, directions)

        # Finally create the LoweredEq with all metadata attached
        expr = super(LoweredEq, cls).__new__(cls, expr.lhs, expr.rhs, evaluate=False)
        expr.is_Increment = getattr(input_expr, 'is_Increment', False)
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

    A SymPy equation carrying its own :class:`IterationSpace`. Suitable for
    use in a :class:`Cluster`.

    Unlike a :class:`LoweredEq`, a ClusterizedEq is "frozen", meaning that any
    call to ``xreplace`` will not trigger re-evaluation (e.g., mathematical
    simplification) of the expression.
    """

    def __new__(cls, *args, **kwargs):
        # Parse input
        if len(args) == 2:
            maybe_ispace = args[1]
            if isinstance(maybe_ispace, IterationSpace):
                input_expr = args[0]
                expr = Eq.__new__(cls, *input_expr.args, evaluate=False)
                expr.is_Increment = input_expr.is_Increment
                expr.ispace = maybe_ispace
            else:
                expr = Eq.__new__(cls, *args, evaluate=False)
                expr.ispace = kwargs['ispace']
                expr.is_Increment = kwargs.get('is_Increment', False)
        else:
            raise ValueError("Cannot construct ClusterizedEq from args=%s "
                             "and kwargs=%s" % (str(args), str(kwargs)))
        return expr

    def func(self, *args, **kwargs):
        return super(ClusterizedEq, self).func(*args, evaluate=False, ispace=self.ispace)

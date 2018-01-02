from collections import OrderedDict

import sympy

from devito.ir.support import Interval, IterationSpace, Stencil
from devito.symbolics import dimension_sort, indexify

__all__ = ['Eq']


class Eq(sympy.Eq):

    """
    A new SymPy equation with an associated iteration space.

    All :class:`Function` objects within ``expr`` get indexified and thus turned
    into objects of type :class:`types.Indexed`.

    An iteration space is an object of type :class:`Space`. It represents the
    data points accessed by the equation along each :class:`Dimension`. The
    :class:`Dimension`s are extracted directly from the equation.
    """

    def __new__(cls, input_expr, subs=None):
        # Sanity check
        assert isinstance(input_expr, sympy.Eq)

        # Indexification
        expr = indexify(input_expr)

        # Apply caller-provided substitution
        if subs is not None:
            expr = expr.xreplace(subs)

        expr = super(Eq, cls).__new__(cls, expr.lhs, expr.rhs, evaluate=False)
        expr.is_Increment = getattr(input_expr, 'is_Increment', False)

        # Get the accessed data points
        stencil = Stencil(expr)

        # Well-defined dimension ordering
        ordering = dimension_sort(expr, key=lambda i: not i.is_Time)

        # Split actual Intervals (the data spaces) from the "derived" iterators,
        # to build an IterationSpace
        iterators = OrderedDict()
        for i in ordering:
            if i.is_Derived:
                iterators.setdefault(i.parent, []).append(stencil.entry(i))
            else:
                iterators.setdefault(i, [])
        intervals = []
        for k, v in iterators.items():
            offs = set.union(set(stencil.get(k)), *[i.ofs for i in v])
            intervals.append(Interval(k, min(offs), max(offs)).negate())
        expr.ispace = IterationSpace(intervals, iterators)

        return expr

    @property
    def is_Scalar(self):
        return self.lhs.is_Symbol

    @property
    def is_Tensor(self):
        return self.lhs.is_Indexed

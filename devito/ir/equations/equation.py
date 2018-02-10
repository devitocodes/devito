from collections import OrderedDict, defaultdict

from sympy import Eq

from devito.dimension import SubDimension
from devito.equation import DOMAIN, INTERIOR
from devito.ir.support import (Interval, DataSpace, IterationSpace, Stencil,
                               IterationInstance, Forward, Backward, Reduction)
from devito.symbolics import dimension_sort, indexify, retrieve_indexed
from devito.tools import flatten

__all__ = ['LoweredEq']


class EqMixin(object):

    """
    A mixin providing operations common to all :mod:`ir` equation types.
    """

    @property
    def is_Scalar(self):
        return self.lhs.is_Symbol

    @property
    def is_Tensor(self):
        return self.lhs.is_Indexed


class LoweredEq(Eq, EqMixin):

    """
    LoweredEq(expr, subs=None)

    A SymPy equation with associated iteration and data spaces.

    All :class:`Function` objects within ``expr`` get indexified and thus turned
    into objects of type :class:`types.Indexed`.

    The data space is an object of type :class:`DataSpace`. It represents the
    data points accessed by the equation along each :class:`Dimension`. The
    :class:`Dimension`s are extracted directly from the equation.

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
            expr.dspace = stamp.dspace
            expr.ispace = stamp.ispace
            return expr
        else:
            raise ValueError("Cannot construct Eq from args=%s "
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

        # Determine the necessary information to build up iteration and data spaces
        intervals, iterators = retrieve_intervals(expr, ordering)
        directions = retrieve_directions(expr, ordering)

        # Finally create the LoweredEq with all metadata attached
        expr = super(LoweredEq, cls).__new__(cls, expr.lhs, expr.rhs, evaluate=False)
        expr.is_Increment = getattr(input_expr, 'is_Increment', False)
        expr.dspace = DataSpace(intervals)
        expr.ispace = IterationSpace([i.negate() for i in intervals],
                                     iterators, directions)

        return expr

    def xreplace(self, rules):
        return LoweredEq(self.lhs.xreplace(rules), self.rhs.xreplace(rules), stamp=self)

    def func(self, *args):
        return super(LoweredEq, self).func(*args, stamp=self, evaluate=False)


def retrieve_intervals(expr, dimensions):
    """Return the data space touched by ``expr``."""
    # Deep retrieval of indexed objects in /expr/
    indexeds = retrieve_indexed(expr, mode='all')
    indexeds += flatten([retrieve_indexed(i) for i in e.indices] for e in indexeds)

    # Detect the indexeds' offsets along each dimension
    stencil = Stencil()
    for e in indexeds:
        for a in e.indices:
            if a in dimensions:
                stencil[a].update([0])
            d = None
            off = [0]
            for i in a.args:
                if i in dimensions:
                    d = i
                elif i.is_integer:
                    off += [int(i)]
            if d is not None:
                stencil[d].update(off)

    # Determine intervals and their iterators
    iterators = OrderedDict()
    for i in dimensions:
        if i.is_NonlinearDerived:
            iterators.setdefault(i.parent, []).append(stencil.entry(i))
        else:
            iterators.setdefault(i, [])
    intervals = []
    for k, v in iterators.items():
        offs = set.union(set(stencil.get(k)), *[i.ofs for i in v])
        intervals.append(Interval(k, min(offs), max(offs)))

    return intervals, iterators


def retrieve_directions(expr, dimensions):
    """Return the directions in which ``dimensions`` must be traversed so
    that information flows when evaluating ``expr``."""
    left, rights = expr.lhs, retrieve_indexed(expr.rhs, mode='all')
    if not left.is_Indexed:
        return []

    # Re-cast as /IterationInstance/s
    left = IterationInstance(left)
    rights = [IterationInstance(i) for i in rights]

    # Determine indexed-wise direction by looking at the vector distance
    mapper = defaultdict(set)
    for i in rights:
        for d in dimensions:
            try:
                distance = left.distance(i, d)
            except TypeError:
                # Nothing can be deduced
                mapper[d].add(Reduction)
                break
            if distance > 0:
                mapper[d].add(Forward)
                break
            elif distance < 0:
                mapper[d].add(Backward)
                break
            mapper[d].add(Reduction)
        # Remainder
        for d in dimensions[dimensions.index(d) + 1:]:
            mapper[d].add(Reduction)
    mapper.update({d.parent: set(mapper[d]) for d in dimensions if d.is_Derived})

    # Resolve clashes. The only illegal case is when Forward and Backward
    # should be used for the same dimension. Mixing Forward/Backward and
    # Reduction is OK, as Forward/Backward win (Reduction implies "arbitrary
    # direction"). When the sole Reduction appears, we default to Forward.
    directions = {}
    for k, v in mapper.items():
        if len(v) == 1:
            direction = v.pop()
            directions[k] = Forward if direction == Reduction else direction
        elif len(v) == 2:
            try:
                v.remove(Reduction)
            except KeyError:
                raise ValueError("Cannot determine flow of equation %s" % expr)
            directions[k] = v.pop()
        else:
            raise ValueError("Cannot determine flow of equation %s" % expr)

    return directions

from devito.finite_differences import IndexDerivative
from devito.ir import Cluster, Interval, IntervalGroup, IterationSpace
from devito.symbolics import uxreplace
from devito.tools import as_tuple, timed_pass
from devito.types import Eq, Inc, Symbol

__all__ = ['lower_index_derivatives']


@timed_pass()
def lower_index_derivatives(clusters, sregistry=None, **kwargs):
    processed = []
    for c in clusters:

        exprs = []
        for e in c.exprs:

            mapper = {}
            for i in e.find(IndexDerivative):
                intervals = [Interval(d, d._min, d._max) for d in i.dimensions]
                ispace0 = IterationSpace(intervals)

                extra = (c.ispace.itdimensions + i.dimensions,)
                ispace = IterationSpace.union(c.ispace, ispace0, relations=extra)

                name = sregistry.make_name(prefix='r')
                s = Symbol(name=name, dtype=e.dtype)
                expr0 = Eq(s, 0.)
                expr1 = Inc(s, i.expr)

                processed.extend([c.rebuild(exprs=expr0),
                                  c.rebuild(exprs=expr1, ispace=ispace)])

                mapper[i] = s

            exprs.append(uxreplace(e, mapper))

        processed.append(c.rebuild(exprs=exprs))

    return processed

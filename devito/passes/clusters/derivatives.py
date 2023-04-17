from devito.finite_differences import IndexDerivative
from devito.ir import Interval, IterationSpace, Queue
from devito.passes.clusters.misc import fuse
from devito.symbolics import (retrieve_dimensions, reuse_if_untouched, q_leaf,
                              uxreplace)
from devito.tools import filter_ordered, timed_pass
from devito.types import Eq, Inc, StencilDimension, Symbol

__all__ = ['lower_index_derivatives']


@timed_pass()
def lower_index_derivatives(clusters, mode=None, **kwargs):
    clusters, weights, mapper = _lower_index_derivatives(clusters, **kwargs)

    if not weights:
        return clusters

    if mode != 'noop':
        clusters = fuse(clusters, toposort='maximal')

    clusters = CDE(mapper).process(clusters)

    return clusters


def _lower_index_derivatives(clusters, sregistry=None, **kwargs):
    weights = {}
    processed = []
    mapper = {}

    def dump(exprs, c):
        if exprs:
            processed.append(c.rebuild(exprs=exprs))
            exprs[:] = []

    for c in clusters:
        exprs = []
        for e in c.exprs:
            expr, v = _core(e, c, weights, mapper, sregistry)
            if v:
                dump(exprs, c)
                processed.extend(v)
            exprs.append(expr)

        dump(exprs, c)

    return processed, weights, mapper


def _core(expr, c, weights, mapper, sregistry):
    """
    Recursively carry out the core of `lower_index_derivatives`.
    """
    if q_leaf(expr):
        return expr, []

    args = []
    processed = []
    for a in expr.args:
        e, clusters = _core(a, c, weights, mapper, sregistry)
        args.append(e)
        processed.extend(clusters)

    expr = reuse_if_untouched(expr, args)

    if not isinstance(expr, IndexDerivative):
        return expr, processed

    # Create concrete Weights and reuse them whenever possible
    name = sregistry.make_name(prefix='w')
    w0 = expr.weights.function
    k = tuple(w0.weights)
    try:
        w = weights[k]
    except KeyError:
        w = weights[k] = w0._rebuild(name=name)
    expr = uxreplace(expr, {w0.indexed: w.indexed})

    dims = retrieve_dimensions(expr, deep=True)
    dims = filter_ordered(d for d in dims if isinstance(d, StencilDimension))

    dims = tuple(reversed(dims))

    # If a StencilDimension already appears in `c.ispace`, perhaps with its custom
    # upper and lower offsets, we honor it
    dims = tuple(d for d in dims if d not in c.ispace)

    intervals = [Interval(d) for d in dims]
    ispace0 = IterationSpace(intervals)

    extra = (c.ispace.itdimensions + dims,)
    ispace = IterationSpace.union(c.ispace, ispace0, relations=extra)

    name = sregistry.make_name(prefix='r')
    s = Symbol(name=name, dtype=c.dtype)
    expr0 = Eq(s, 0.)
    ispace1 = ispace.project(lambda d: d is not dims[-1])
    processed.insert(0, c.rebuild(exprs=expr0, ispace=ispace1))

    # Transform e.g. `w[i0] -> w[i0 + 2]` for alignment with the
    # StencilDimensions starting points
    subs = {expr.weights: expr.weights.subs(d, d - d._min) for d in dims}
    expr1 = Inc(s, uxreplace(expr.expr, subs))
    processed.append(c.rebuild(exprs=expr1, ispace=ispace))

    # Track lowered IndexDerivative for subsequent optimization by the caller
    mapper.setdefault(expr1.rhs, []).append(s)

    return s, processed


class CDE(Queue):

    """
    Common derivative elimination.
    """

    def __init__(self, mapper):
        super().__init__()

        self.mapper = {k: v for k, v in mapper.items() if len(v) > 1}

    def process(self, clusters):
        return self._process_fdta(clusters, 1, subs0={}, seen=set())

    def callback(self, clusters, prefix, subs0=None, seen=None):
        subs = {}
        processed = []
        for c in clusters:
            if c in seen:
                processed.append(c)
                continue

            exprs = []
            for e in c.exprs:
                k, v = e.args

                if k in subs0:
                    continue

                try:
                    subs0[k] = subs[v]
                    continue
                except KeyError:
                    pass

                if v in self.mapper:
                    subs[v] = k
                    exprs.append(e)
                else:
                    exprs.append(uxreplace(e, {**subs0, **subs}))

            processed.append(c.rebuild(exprs=exprs))

        seen.update(processed)

        return processed

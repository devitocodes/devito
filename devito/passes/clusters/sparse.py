from functools import partial

from devito.ir import ClusterizedEq, Interval, IterationSpace
from devito.symbolics import uxreplace
from devito.tools import timed_pass
from devito.types import Eq, Inc, Temp

__all__ = ['lower_sparse_sum', 'lower_sparse_sums']


@timed_pass()
def lower_sparse_sums(clusters, sregistry=None, **kwargs):
    """
    Lower SparseLocalSums into a private initializer and guarded accumulation.

    For example, using the interpolation's radius dimension and guard::

        i = CustomDimension('i', 0, 1, 2)
        ci = ConditionalDimension('i', i, indirect=True,
            condition=And(pos + i >= x_m, pos + i <= x_M))
        Eq(rcv[p], SparseLocalSum(w[p, ci]*f[pos + ci], (ci,), f.dtype))

    becomes the following computation at each sparse point `p` (pseudocode)::

        sum0 = 0
        for i in range(2):
            if x_m <= pos + i <= x_M:
                sum0 += w[p, i]*f[pos + i]
        rcv[p] = sum0

    The initializer and result stay outside the tap guard, so even a fully
    masked stencil assigns zero to `rcv[p]`.
    """
    processed = []
    for c in clusters:
        if not c.sparse_sums:
            processed.append(c)
            continue

        for e in c.exprs:
            subs = {}
            for reduction in e.sparse_sums:
                init, update, value = lower_sparse_sum(
                    c, uxreplace(reduction, subs), sregistry
                )
                processed.extend([init, update])
                subs[reduction] = value

            expr = e.apply(partial(uxreplace, rule=subs))
            processed.append(c.rebuild(exprs=[expr]))

    return processed


def lower_sparse_sum(cluster, reduction, sregistry):
    """
    Construct the private initializer and guarded accumulation for one sum.
    """
    value = Temp(name=sregistry.make_name(prefix='sum'), dtype=reduction.dtype)

    dims = reduction.dimensions
    inner = IterationSpace([Interval(d) for d in dims])
    ispace = IterationSpace.union(
        cluster.ispace, inner, relations=(cluster.ispace.itdims + dims,)
    )

    init = cluster.rebuild(exprs=[Eq(value, 0)])
    # Attach the interpolation's original guards to the accumulation only
    expr = ClusterizedEq(
        Inc(value, reduction.expr), ispace=ispace, conditionals=reduction.conditionals
    )

    properties = cluster.properties.sequentialize(dims)

    update = cluster.rebuild(exprs=expr, ispace=ispace, properties=properties)

    return init, update, value

from devito.ir import ClusterizedEq, Interval, IterationSpace
from devito.symbolics import uxreplace
from devito.tools import timed_pass
from devito.types import Eq, Inc, Temp

__all__ = ['lower_local_sum', 'lower_local_sums']


@timed_pass()
def lower_local_sums(clusters, sregistry=None, **kwargs):
    """
    Lower LocalSums into private initializers and guarded accumulations.

    For example, using the interpolation's radius dimension and guard::

        i = CustomDimension('i', 0, 1, 2)
        ci = ConditionalDimension('i', i, indirect=True,
            condition=And(pos + i >= x_m, pos + i <= x_M))
        Eq(rcv[p], LocalSum(w[p, ci]*f[pos + ci], cdims=(ci,)))

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
        if not c.local_sums:
            processed.append(c)
            continue

        for e in c.exprs:
            subs = {}
            for reduction in e.local_sums:
                init, update, value = lower_local_sum(
                    c, uxreplace(reduction, subs), sregistry
                )
                processed.extend([init, update])
                subs[reduction] = value

            expr = uxreplace(e, subs)
            processed.append(c.rebuild(exprs=[expr]))

    return processed


def lower_local_sum(cluster, reduction, sregistry):
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

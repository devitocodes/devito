from devito.ir.clusters import Queue
from devito.ir.support import GuardFactor
from devito.symbolics import IntDiv, uxreplace
from devito.tools import timed_pass

__all__ = ['subsampling']


@timed_pass()
def subsampling(clusters):
    """
    Strengthen factor-based ConditionalDimension guards into strided iteration.

    A Function subsampled by `factor` along `d` is written under a
    `d % factor == 0` guard, so the enclosing loop visits every point of `d`
    and discards all but one in `factor` of them. Where legal, drop the guard
    and step the loop `factor` points at a time instead

        for (x = x_m; x <= x_M; x += 1)      for (x = ...; x <= x_M; x += xf)
          if (x%xf == 0)                 ->     g[x/xf] = ...
            g[x/xf] = ...

The loop is taken over the subsampled Dimension `cd` itself, with a *unit*
    stride, and every access to the parent `d` is rescaled to `cd*factor`:

        for (x = x_m; x <= x_M; x += 1)      for (xs = xs_m; xs <= xs_M; xs++)
          if (x%xf == 0)                 ->     g[xs] = u[xs*xf]
            g[x/xf] = u[x]

    A non-unitary increment is deliberately avoided: it would escape the
    Lamport-style dependence analysis the compiler relies on. The loop bounds
    are `cd`'s own runtime values, which `ConditionalDimension._arg_values`
    derives as `ceil(x_m/factor) .. floor(x_M/factor)`, so no anchoring of the
    loop minimum is needed either.

    Parameters
    ----------
    clusters : tuple of Clusters
        Input Clusters, subject of the optimization pass.
    """
    return Subsampling().process(clusters)


class Subsampling(Queue):

    """
    Replace uniform factor-based guards with strided iteration.

    Clusters are grouped by iteration space prefix alone, deliberately *not*
    by guard: a guard may only become a stride if every Cluster sharing the
    prefix carries the very same one. Striding a loop nest that also hosts an
    unguarded expression would run that expression on a subset of its
    iterations -- and dropping the guard is what makes the nest shared in the
    first place, since `stree_build` reuses a subtree across Clusters with
    matching guards (see `reuse_whole_subtree`).
    """

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        guards = {c.guards.get(d) for c in clusters}
        if len(guards) > 1:
            # Non-uniformly guarded, hence `d`'s loop nest is shared with
            # expressions that must keep running at every point
            return clusters

        guard = guards.pop()
        if not all(_is_strideable(c, d, guard) for c in clusters):
            return clusters

        cd = guard.d
        factor = cd.symbolic_factor

        processed = []
        for c in clusters:
            # Iterate the subsampled Dimension itself, with unit stride, and
            # rescale every parent access `d` to `cd*factor`
            ispace = c.ispace.switch(d, cd)

            # `cd`'s parent index is `cd*p//q`, and the lowered access
            # `IntDiv(q*d, p)` collapses to `cd` under it
            index = IntDiv(cd*factor, cd.factor_den)
            subs = {IntDiv(cd.factor_den*d, factor): cd, d: index}
            exprs = [uxreplace(e, subs) for e in c.exprs]

            # `cd` inherits `d`'s computational properties: it covers the very
            # same points, just addressed in subsampled coordinates. Without
            # this the loop would lose PARALLEL and so all of its OpenMP
            properties = c.properties.drop(d)
            properties = properties.add(cd, c.properties[d])

            processed.append(c.rebuild(exprs=exprs, ispace=ispace,
                                       guards=c.guards.popany(d),
                                       properties=properties))

        return processed


def _is_strideable(cluster, d, guard):
    """
    True if iteration over `d` may step `guard`'s subsampling factor points at
    a time in place of evaluating `guard` at every point.
    """
    # The guard must be exactly a subsampling guard. Anything `And`/`Or`-ed in
    # (e.g. a user-supplied `condition`) must still be evaluated pointwise
    if not isinstance(guard, GuardFactor):
        return False

    if d not in cluster.ispace.itdims:
        return False

    # Skipping points is only sound if no dependence is carried along `d`;
    # otherwise the discarded iterations may be doing work that `d`'s
    # remaining iterations rely upon
    if not cluster.properties.is_parallel_relaxed(d):
        return False

    # Non-affine accesses may read the points that would be skipped
    return cluster.properties.is_affine(d)

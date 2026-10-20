from devito.ir.clusters import Queue
from devito.ir.support import GuardFactor, Strided
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

    The rewrite is recorded as a `Strided` property on `d`; the stride and the
    anchoring of the loop min are applied by `NodeIteration.limits`. The
    iteration space itself is left keyed on `d`, so the runtime `x_m`/`x_M`
    values, the domain decomposition and any non-subsampled access `f[x + 1]`
    are all unaffected.

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

        processed = []
        for c in clusters:
            properties = c.properties.add(d, Strided(guard.d.symbolic_factor))
            processed.append(c.rebuild(guards=c.guards.popany(d),
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

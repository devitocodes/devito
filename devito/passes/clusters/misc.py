from itertools import groupby, product

from devito.ir.clusters import Queue, cluster_pass
from devito.ir.support import SEPARABLE, SEQUENTIAL, Scope
from devito.passes.clusters.utils import in_critical_region
from devito.symbolics import pow_to_mul
from devito.tools import Stamp, flatten, frozendict, timed_pass
from devito.types import Hyperplane

__all__ = ['Lift', 'fission', 'optimize_hyperplanes', 'optimize_pows']


class Lift(Queue):

    """
    Remove invariant Dimensions from Clusters to avoid redundant computation.

    Notes
    -----
    This is analogous to the compiler transformation known as
    "loop-invariant code motion".
    """

    @timed_pass(name='lift')
    def process(self, elements):
        return super().process(elements)

    def callback(self, clusters, prefix):
        if not prefix:
            # No iteration space to be lifted from
            return clusters
        dim = prefix[-1].dim

        hope_invariant = dim._defines
        outer = set().union(*[i.dim._defines for i in prefix[:-1]])

        lifted = []
        processed = []
        for n, c in enumerate(clusters):
            # Storage-related dependences, such as those induced by reduction
            # increments, prevent lifting
            if any(dep.is_storage_related(dim) for dep in c.scope.d_all_gen()):
                processed.append(c)
                continue

            # Synchronization prevents lifting
            if any(c.syncs.get(d) for d in dim._defines) or \
               in_critical_region(c, clusters):
                processed.append(c)
                continue

            # Is `c` a real candidate -- is there at least one invariant Dimension?
            if any(d._defines & hope_invariant for d in c.exprs_dimensions):
                processed.append(c)
                continue

            impacted = set(processed) | set(clusters[n+1:])

            # None of the Functions appearing in a lifted Cluster can be written to
            if any(c.functions & set(i.scope.writes) for i in impacted):
                processed.append(c)
                continue

            # All of the inner Dimensions must appear in the write-to region
            # otherwise we would violate data dependencies. Consider
            #
            # 1)                2)                        3)
            # for i             for i                     for i
            #   for x             for x                     for x
            #     r = f(a[x])       for y                     for y
            #                         r[x] = f(a[x, y])         r[x, y] = f(a[x, y])
            #
            # In 1) and 2) lifting is infeasible; in 3) the statement can
            # be lifted outside the `i` loop as `r`'s write-to region contains
            # both `x` and `y`
            xed = {d._defines for d in c.exprs_dimensions if d not in outer}
            if not all(i & set(w.dimensions) for i, w in product(xed, c.scope.writes)):
                processed.append(c)
                continue

            # The contracted iteration and data spaces
            key = lambda d: d not in hope_invariant
            ispace = c.ispace.project(key)

            # Optimization: if not lifting from the innermost Dimension, we can
            # safely reset the `ispace` to expose potential fusion opportunities
            try:
                if c.ispace.innermost.dim not in hope_invariant:
                    ispace = ispace.reset()
            except IndexError:
                pass

            properties = c.properties.filter(key)

            # If `c` is made of scalar expressions within guards, then we must keep
            # it close to the adjacent Clusters for correctness
            if c.is_scalar and c.guards and ispace:
                processed.append(c.rebuild(ispace=ispace, properties=properties))
            else:
                lifted.append(c.rebuild(ispace=ispace, properties=properties))

        return lifted + processed


@cluster_pass(mode='all')
def optimize_pows(cluster, *args):
    """
    Convert integer powers into Muls, such as ``a**2 => a*a``.
    """
    return cluster.rebuild(exprs=pow_to_mul(cluster.exprs))


class Fission(Queue):

    """
    Implement Clusters fission. For more info refer to fission.__doc__.
    """

    def callback(self, clusters, prefix):
        if not prefix or len(clusters) == 1:
            return clusters

        d = prefix[-1].dim

        # Do not waste time if definitely illegal
        if any(SEQUENTIAL in c.properties[d] for c in clusters):
            return clusters

        # Do not waste time if definitely nothing to do
        if all(len(prefix) == len(c.ispace) for c in clusters):
            return clusters

        # Analyze and abort if fissioning would break a dependence
        scope = Scope(flatten(c.exprs for c in clusters))
        if any(d._defines & dep.cause or dep.is_reduce(d) for dep in scope.d_all_gen()):
            return clusters

        processed = []
        for (it, guards), g in groupby(clusters, key=lambda c: self._key(c, prefix)):
            group = list(g)

            try:
                test0 = any(SEQUENTIAL in c.properties[it.dim] for c in group)
            except AttributeError:
                # `it` is None because `c`'s IterationSpace has no `d` Dimension,
                # hence `key = (it, guards) = (None, guards)`
                test0 = True

            if test0 or guards:
                # Heuristic: no gain from fissioning if unable to ultimately
                # increase the number of collapsible iteration spaces, hence give up
                processed.extend(group)
            else:
                stamp = Stamp()
                for c in group:
                    ispace = c.ispace.lift(d, stamp)
                    processed.append(c.rebuild(ispace=ispace))

        return processed

    def _key(self, c, prefix):
        try:
            index = len(prefix)
            dims = tuple(i.dim for i in prefix)

            it = c.ispace[index]
            guards = frozendict({d: v for d, v in c.guards.items() if d in dims})

            return (it, guards)
        except IndexError:
            return (None, c.guards)


@timed_pass()
def fission(clusters):
    """
    Clusters fission.

    Currently performed in the following cases:

        * Trade off data locality for parallelism, e.g.

          .. code-block::

            for x              for x
              for y1             for y1
                ..                 ..
              for y2     -->   for x
                ..               for y2
                                   ..
    """
    return Fission().process(clusters)


@timed_pass()
def optimize_hyperplanes(clusters):
    """
    At the moment this is just a dummy no-op pass that we only use
    for testing purposes.
    """
    for c in clusters:
        for k, v in c.properties.items():
            if isinstance(k, Hyperplane) and SEPARABLE in v:
                raise NotImplementedError

    return clusters

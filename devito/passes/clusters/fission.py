from itertools import groupby

from devito.ir import Queue, Scope
from devito.tools import Stamp, flatten, frozendict, timed_pass

__all__ = ['fission']


class FissionForParallelism(Queue):

    def callback(self, clusters, prefix):
        if not prefix or len(clusters) == 1:
            return clusters

        d = prefix[-1].dim

        # Do not waste time if definitely illegal
        if any(c.properties.is_sequential(d) for c in clusters):
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
                test0 = any(c.properties.is_sequential(it.dim) for c in group)
            except AttributeError:
                # `it` is None because `c`'s IterationSpace has no `d` Dimension,
                # hence `key = (it, guards) = (None, guards)`
                test0 = True

            if test0 or guards:
                # Heuristic: no gain from fissioning if unable to ultimately
                # increase the number of collapsable iteration spaces, hence give up
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


class FissionForPressure(Queue):

    def callback(self, clusters, prefix):
        if not prefix or len(clusters) == 1:
            return clusters

        d = prefix[-1].dim

        from IPython import embed; embed()

        return clusters


@timed_pass()
def fission(clusters, kind='parallelism', **kwargs):
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

        * Trade off data locality for register pressure, e.g.

          .. code-block::

            for x                         for x
              for y                         for y1
                a = f(x) + g(x)                 a = f(x) + g(x)
                b = h(x) + w(x)     -->     for y2
                                                b = h(x) + w(x)
    """
    assert kind in ('parallelism', 'pressure', 'all')

    if kind in ('parallelism', 'all'):
        clusters = FissionForParallelism().process(clusters)

    if kind in ('pressure', 'all'):
        clusters = FissionForPressure().process(clusters)

    return clusters

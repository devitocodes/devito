from itertools import groupby

from devito.ir import Queue, Scope
from devito.symbolics import retrieve_terminals
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


def fission_for_pressure(clusters, options):
    fiss_press_ratio = options['fiss-press-ratio']
    fiss_press_size = options['fiss-press-size']

    processed = []
    for c in clusters:
        if not c.ispace:
            processed.append(c)
            continue

        # Fission, if anything, occurs along the innermost Dimension
        d = c.ispace[-1].dim

        # Let `ts` ("timestamp") be our candidate split point
        for timestamp in range(1, len(c.exprs)):
            # Checking whether it's legal or not might be expensive, so let's
            # first find out whether it'd be worth it
            g0 = c.exprs[:timestamp]
            g1 = c.exprs[timestamp:]

            terminals0 = retrieve_terminals(g0, mode='unique')
            if len(terminals0) < fiss_press_size:
                continue
            terminals1 = retrieve_terminals(g1, mode='unique')
            if len(terminals1) < fiss_press_size:
                continue

            functions0 = {i.function for i in terminals0 if i.is_Indexed}
            functions1 = {i.function for i in terminals1 if i.is_Indexed}
            functions_shared = functions0.intersection(functions1)

            n0 = len(functions0)
            n1 = len(functions1)
            ns = len(functions_shared)

            if not ns:
                ns = .001

            if not (n0 / ns >= fiss_press_ratio and n1 / ns >= fiss_press_ratio):
                continue

            # At this point we know we want to fission. But can we?
            for dep in c.scope.d_flow.independent():
                if dep.source.timestamp < timestamp <= dep.sink.timestamp:
                    # Nope, we would unfortunately violate a data dependence
                    break
            else:
                # Yes -- all good
                processed.append(c.rebuild(exprs=g0))

                ispace = c.ispace.lift(d)
                processed.append(c.rebuild(exprs=g1, ispace=ispace))

                break
        else:
            processed.append(c)

    return processed


@timed_pass()
def fission(clusters, kind='parallelism', options=None, **kwargs):
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

          NOTE: this only applies to innermost Dimensions.
    """
    assert kind in ('parallelism', 'pressure', 'all')

    if kind in ('parallelism', 'all'):
        clusters = FissionForParallelism().process(clusters)

    if kind in ('pressure', 'all'):
        clusters = fission_for_pressure(clusters, options)

    return clusters

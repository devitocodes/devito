from collections import OrderedDict, namedtuple

from devito.ir.support.basic import Scope
from devito.ir.clusters.cluster import Cluster
from devito.ir.dfg import TemporariesGraph
from devito.ir.support import Stencil
from devito.symbolics import xreplace_indices
from devito.types import Scalar
from devito.tools import flatten

__all__ = ['clusterize', 'merge']


def merge(clusters):
    """
    Given an ordered collection of :class:`Cluster` objects, return a
    (potentially) smaller sequence in which clusters with identical stencil
    have been merged into a single :class:`Cluster`, as long as data
    dependences are honored.
    """

    def _merge(source, sink):
        temporaries = OrderedDict(source.trace.items())
        temporaries.update(OrderedDict([(k, v) for k, v in sink.trace.items()
                                        if k not in temporaries]))
        return Cluster(temporaries.values(), source.stencil, source.atomics)

    processed = []
    for c in clusters:
        merged = None
        for candidate in reversed(list(processed)):
            # Get the data dependences relevant for cluster fusion
            scope = Scope(exprs=candidate.exprs + c.exprs)
            d_true = scope.d_anti.carried() - scope.d_anti.indirect()
            if candidate.stencil == c.stencil:
                d_fake = scope.d_flow.independent() - scope.d_flow.indirect()
                d_funcs = [i.function for i in d_true + d_fake]
                if all(is_local(i, candidate, c, clusters) for i in d_funcs):
                    # /c/ will be merged into /candidate/. However, all fusion-induced
                    # anti dependences need to be removed. This is achieved through so
                    # called index bumping and array contraction
                    merged = _merge(*bump_and_contract(d_funcs, candidate, c))
                    processed[processed.index(candidate)] = merged
                break
            elif scope.d_all:
                # Data dependences prevent fusion with earlier Clusters.
                break
        # Fallback
        if not merged:
            processed.append(c)

    return processed


def is_local(array, source, sink, context):
    """
    Return True if ``array`` satisfies the following conditions: ::

        * it's a temporary; that is, of type :class:`Array`;
        * it's written once, within the ``source`` :class:`Cluster`;
        * it's read in the ``sink`` :class:`Cluster` only; in particular,
          it doesn't appear in any other :class:`Cluster`s out of those
          provided in ``context``.

    If any of these conditions do not hold, return False.
    """
    if not array.is_Array:
        return False

    # Written in source
    if array not in [i.function for i in source.trace.values()]:
        return False

    # Never read outside of sink
    context = [i for i in context if i not in [source, sink]]
    if array in flatten(i.unknown for i in context):
        return False

    return True


def bump_and_contract(targets, source, sink):
    """
    Return a new source and a new sink in which the :class:`Array`s in ``targets``
    have been turned into scalars. This is implemented through index bumping and
    array contraction.

    :param targets: The :class:`Array`s that will be contracted.
    :param source: The source :class:`Cluster`.
    :param sink: The sink :class:`Cluster`.

    Examples
    ========
    Index bumping
    -------------
    Given: ::

        r[x,y,z] = b[x,y,z]*2

    Produce: ::

        r[x,y,z] = b[x,y,z]*2
        r[x,y,z+1] = b[x,y,z+1]*2

    Array contraction
    -----------------
    Given: ::

        r[x,y,z] = b[x,y,z]*2
        r[x,y,z+1] = b[x,y,z+1]*2

    Produce: ::

        tmp0 = b[x,y,z]*2
        tmp1 = b[x,y,z+1]*2

    Full example (bump+contraction)
    -------------------------------
    Given: ::

        source: [r[x,y,z] = b[x,y,z]*2]
        sink: [a = ... r[x,y,z] ... r[x,y,z+1] ...]
        targets: r

    Produce: ::

        source: [tmp0 = b[x,y,z]*2, tmp1 = b[x,y,z+1]*2]
        sink: [a = ... tmp0 ... tmp1 ...]
    """
    if not targets:
        return source, sink

    mapper = {}

    # Build the new source
    processed = []
    for k, v in source.trace.items():
        if any(v.function not in i for i in [targets, sink.tensors]):
            processed.append(v.func(k, v.rhs.xreplace(mapper)))
        else:
            for i in sink.tensors[v.function]:
                scalarized = Scalar(name='s%d' % len(mapper)).indexify()
                mapper[i] = scalarized

                # Index bumping
                assert len(v.function.indices) == len(k.indices) == len(i.indices)
                shifting = {idx: idx + (o2 - o1) for idx, o1, o2 in
                            zip(v.function.indices, k.indices, i.indices)}

                # Array contraction
                handle = v.func(scalarized, v.rhs.xreplace(mapper))
                handle = xreplace_indices(handle, shifting)
                processed.append(handle)
    source = source.rebuild(processed)

    # Build the new sink
    processed = [v.func(k, v.rhs.xreplace(mapper)) for k, v in sink.trace.items()]
    sink = sink.rebuild(processed)

    return source, sink


def aggregate(exprs, stencils):
    """
    Aggregate consecutive expressions having identical LHS and identical
    stencil by substituting the earlier with the later ones, in program order.

    Examples
    ========
    a[i] = b[i] + 3.
    a[i] = c[i] + 4. + a[i]
    >> a[i] = c[i] + 4. + b[i] + 3.
    """
    mapper = OrderedDict(zip(exprs, stencils))

    groups = []
    last = None
    for k, v in mapper.items():
        key = k.lhs, v
        if key == last:
            groups[-1].append(k)
        else:
            last = key
            groups.append([k])

    exprs, stencils = [], []
    for i in groups:
        top = i[0]
        if len(i) == 1:
            exprs.append(top)
        else:
            inlining, base = top.args
            queue = list(i[1:])
            while queue:
                base = queue.pop(0).rhs.xreplace({inlining: base})
            exprs.append(top.func(inlining, base))
        stencils.append(mapper[top])

    return exprs, stencils


def clusterize(exprs, stencils):
    """
    Derive :class:`Cluster` objects from an iterable of expressions; a stencil for
    each expression must be provided.
    """
    assert len(exprs) == len(stencils)

    exprs, stencils = aggregate(exprs, stencils)

    Info = namedtuple('Info', 'trace stencil')

    # Build a dependence graph and associate each node with its Stencil
    mapper = OrderedDict()
    g = TemporariesGraph(exprs)
    for (k, v), j in zip(g.items(), stencils):
        if v.is_tensor:
            trace = g.trace(k)
            trace += tuple(i for i in g.trace(k, readby=True) if i not in trace)
            mapper[k] = Info(trace, j)

    # A cluster stencil is determined iteratively, by first calculating the
    # "local" stencil and then by looking at the stencils of all other clusters
    # depending on it. The stencil information is propagated until there are
    # no more updates
    queue = list(mapper)
    while queue:
        target = queue.pop(0)

        info = mapper[target]
        strict_trace = [i.lhs for i in info.trace if i.lhs != target]

        stencil = Stencil(info.stencil.entries)
        for i in strict_trace:
            if i in mapper:
                stencil = stencil.add(mapper[i].stencil)

        mapper[target] = Info(info.trace, stencil)

        if stencil != info.stencil:
            # Something has changed, need to propagate the update
            queue.extend([i for i in strict_trace if i not in queue])

    clusters = []
    for target, info in mapper.items():
        # Drop all non-output tensors, as computed by other clusters
        exprs = [i for i in info.trace if i.lhs.is_Symbol or i.lhs == target]

        # Create and track the cluster
        clusters.append(Cluster(exprs, info.stencil.frozen))

    clusters = merge(clusters)

    # For each cluster, derive its atomics dimensions
    for c1 in clusters:
        atomics = set()
        for c2 in reversed(clusters[:clusters.index(c1)]):
            scope = Scope(exprs=c1.exprs + c2.exprs)
            true = scope.d_anti.carried() - scope.d_anti.indirect()
            atomics |= set(c1.stencil.dimensions) & set(true.cause)
        c1.atomics = atomics

    return clusters

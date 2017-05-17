from collections import OrderedDict

from devito.dse.graph import temporaries_graph

from devito.stencil import Stencil

__all__ = ['clusterize']


class Cluster(object):

    """
    A Cluster is an ordered sequence of expressions that are necessary to
    compute a tensor, plus the tensor expression itself.

    A Cluster is associated with a stencil, which tracks what neighboring points
    are required, along each dimension, to compute an entry in the tensor.
    """

    def __init__(self, exprs, stencil):
        self.trace = temporaries_graph(exprs)
        self.stencil = stencil

    @property
    def exprs(self):
        return self.trace.values()

    @property
    def is_dense(self):
        return self.trace.space_indices and not self.trace.time_invariant()

    @property
    def is_sparse(self):
        return not self.is_dense

    def rebuild(self, exprs):
        return Cluster(exprs, self.stencil)


def merge(clusters):
    """
    Given an ordered collection of :class:`Cluster` objects, return a
    (potentially) smaller sequence in which clusters with identical stencil
    have been merged into a single :class:`Cluster`.
    """
    mapper = OrderedDict()
    for c in clusters:
        mapper.setdefault(c.stencil.entries, []).append(c)

    processed = []
    for entries, clusters in mapper.items():
        # Eliminate redundant temporaries
        temporaries = OrderedDict()
        for c in clusters:
            for k, v in c.trace.items():
                if k not in temporaries:
                    temporaries[k] = v
        # Squash the clusters together
        processed.append(Cluster(temporaries.values(), Stencil(entries)))

    return processed


def aggregate(exprs, stencils):
    """
    Consecutive expressions with identical LHS and identical stencil
    may be aggregated by substituting the earlier with the later ones (in
    program order).

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
    """Derive :class:`Cluster`s from an iterator of expressions; a stencil for
    each expression must be provided."""
    assert len(exprs) == len(stencils)

    exprs, stencils = aggregate(exprs, stencils)

    g = temporaries_graph(exprs)
    mapper = OrderedDict([(i.lhs, j) for i, j in zip(g.values(), stencils)
                          if i.is_tensor])

    clusters = []
    for k, v in mapper.items():
        # Determine what temporaries are needed to compute /i/
        exprs = g.trace(k)

        # Determine the Stencil of the cluster
        stencil = Stencil(v.entries)
        for i in exprs:
            stencil = stencil.add(mapper.get(i.lhs, {}))
        stencil = stencil.frozen

        # Drop all non-output tensors, as computed by other clusters
        exprs = [i for i in exprs if i.lhs.is_Symbol or i.lhs == k]

        # Create and track the cluster
        clusters.append(Cluster(exprs, stencil))

    return merge(clusters)

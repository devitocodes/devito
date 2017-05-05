from collections import OrderedDict

from devito.dse.graph import temporaries_graph

from devito.stencil import Stencil


class BasicCluster(object):

    """
    A BasicCluster is an ordered sequence of expressions that are necessary to
    compute a tensor, plus the tensor expression itself.

    A BasicCluster is associated with a stencil, which tracks what neighboring points
    are required, along each dimension, to compute an entry in the tensor.

    Examples
    ========
    In the following sequence of operations: ::

        temp1 = a*b
        temp2 = c
        temp3[k] = temp1 + temp2
        temp4[k] = temp2 + 5
        temp5 = d*e
        temp6 = f+g
        temp7[i] = temp5 + temp6 + temp4[k]

    There are three target expressions: temp3, temp4, temp7. There are therefore
    three BasicClusters: ((temp1, temp2, temp3), (temp2, temp4), (temp5, temp6, temp7)).
    The first and the second share the expression temp2. Note that temp4 does
    not appear in the third BasicCluster, as it is not a scalar.
    """

    def __init__(self, exprs, stencil):
        self.trace = temporaries_graph(exprs)
        self.stencil = stencil

        self.output = exprs[-1]


class Cluster(object):

    """
    A Cluster is obtained by merging an ordered collection of :class:`BasicCluster`
    having identical stencil, so it has more than one output tensors.
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
    Given an ordered collection of :class:`BasicCluster` objects, return a
    (potentially) smaller sequence in which clusters with identical stencil
    have been merged into a :class:`Cluster`.
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


def clusterize(exprs, stencils):
    """Compute :class:`Cluster`s from an iterator of expressions."""
    graph = temporaries_graph(exprs)
    targets = [k for k, v in graph.items() if v.is_tensor]

    clusters = []
    for i in targets:
        # Determine what temporaries are needed to compute /i/
        exprs = graph.trace(i)

        # Determine the Stencil of the cluster
        stencil = Stencil(stencils[i].entries)
        for j in exprs:
            if j.lhs in stencils:
                stencil = stencil.add(stencils[j.lhs])
        stencil = stencil.frozen

        # Drop all non-output tensors, as computed by other clusters
        exprs = [j for j in exprs if j.lhs.is_Symbol or j.lhs == i]

        # Create and track the cluster
        clusters.append(Cluster(exprs, stencil))

    return merge(clusters)

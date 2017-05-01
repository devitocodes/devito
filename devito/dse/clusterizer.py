from collections import OrderedDict

from devito.dse.graph import temporaries_graph

from devito.stencil import Stencil


class Cluster(object):

    """
    A Cluster is an ordered sequence of expressions that are necessary to
    compute a tensor, plus the tensor expression itself.

    A Cluster is associated with a stencil, which tracks what neighboring points
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
    three clusters: ((temp1, temp2, temp3), (temp2, temp4), (temp5, temp6, temp7)).
    The first and second clusters share the expression temp2. Note that temp4 does
    not appear in the third cluster, as it is not a scalar.
    """

    def __init__(self, trace, stencils):
        self._full_trace = trace
        self._output = trace.values()[-1]

        # Compute the stencil of the cluster
        stencil = Stencil(stencils[self.output.lhs].entries)
        for i in trace:
            if i in stencils:
                stencil = stencil.add(stencils[i])
        self._stencil = stencil.frozen

    def _view(self, drop=lambda v: False):
        cls = type(self._full_trace)
        return cls([(k, v) for k, v in self._full_trace.items() if not drop(v)])

    @property
    def output(self):
        return self._output

    @property
    def trace(self):
        """The ordered collection of expressions to compute the output tensor."""
        return self._view(drop=lambda v: v.is_tensor and v != self.output)

    @property
    def stencil(self):
        return self._stencil


class SuperCluster(object):

    """
    A SuperCluster represents the result of merging a collection of cluster.
    """

    @classmethod
    def merge(cls, clusters):
        """
        Given an ordered collection of :class:`Cluster` objects, return a
        (potentially) smaller sequence in which clusters with identical stencil
        have been merged.
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
            supertrace = temporaries_graph(temporaries.values())
            processed.append(SuperCluster(supertrace, Stencil(entries)))

        return processed

    def __init__(self, trace, stencil):
        self.trace = trace
        self.stencil = stencil


def clusterize(exprs, stencils):
    """Compute :class:`Cluster`s from an iterator of expressions."""
    graph = temporaries_graph(exprs)
    targets = [k for k, v in graph.items() if v.is_tensor]

    clusters = []
    for i in targets:
        # Determine what temporaries are needed to compute /i/
        trace = graph.trace(i)
        # Create and track the cluster
        clusters.append(Cluster(trace, stencils))

    return SuperCluster.merge(clusters)

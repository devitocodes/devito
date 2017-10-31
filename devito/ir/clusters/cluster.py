from devito.ir.dfg import TemporariesGraph
from devito.tools import as_tuple


class Cluster(object):

    """
    A Cluster is an ordered sequence of expressions that are necessary to
    compute a tensor, plus the tensor expression itself.

    A Cluster is associated with a stencil, which tracks what neighboring points
    are required, along each dimension, to compute an entry in the tensor.

    The parameter ``atomics`` allows to specify dimensions (a subset of those
    appearing in ``stencil``) along which a Cluster cannot be fused with
    other clusters. This is for example useful when a Cluster is evaluating
    a tensor temporary, whose values must all be updated before being accessed
    in the subsequent clusters.
    """

    def __init__(self, exprs, stencil, atomics=None):
        self.trace = TemporariesGraph(exprs)
        self.stencil = stencil
        self._atomics = as_tuple(atomics)

    @property
    def exprs(self):
        return list(self.trace.values())

    @property
    def unknown(self):
        return self.trace.unknown

    @property
    def tensors(self):
        return self.trace.tensors

    @property
    def is_dense(self):
        return self.trace.space_indices and not self.trace.time_invariant()

    @property
    def is_sparse(self):
        return not self.is_dense

    @property
    def atomics(self):
        return self._atomics

    @atomics.setter
    def atomics(self, val):
        self._atomics = as_tuple(val)

    def rebuild(self, exprs):
        """
        Build a new cluster with expressions ``exprs`` having same stencil as ``self``.
        """
        return Cluster(exprs, self.stencil, self.atomics)

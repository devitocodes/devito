from devito.ir import FindNodes
from devito.targets import Intel64Rewriter, Ompizer

from devito.yask.utils import Offloaded

__all__ = ['YaskRewriter']


class YaskOmpizer(Ompizer):

    def __init__(self, key=None):
        if key is None:
            def key(i):
                # If it's not parallel, nothing to do
                if not i.is_ParallelRelaxed or i.is_Vectorizable:
                    return False
                # If some of the inner computation has been offloaded to YASK,
                # avoid introducing an outer level of parallelism
                if FindNodes(Offloaded).visit(i):
                    return False
                return True
        super(YaskOmpizer, self).__init__(key=key)


class YaskRewriter(Intel64Rewriter):

    _parallelizer_shm_type = YaskOmpizer

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._loop_wrapping(state)
        if self.params['openmp']:
            self._parallelize_shm(state)

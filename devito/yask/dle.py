from devito.ir import FindNodes
from devito.targets import (Intel64Rewriter, Ompizer, avoid_denormals, loop_wrapping,
                            parallelize_shm, insert_defs, insert_casts)

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
        # Optimization and parallelism
        avoid_denormals(state)
        loop_wrapping(state)
        if self.params['openmp']:
            parallelize_shm(state, parallelizer_shm=self.parallelizer_shm)

        # Symbol definitions
        insert_defs(state)
        insert_casts(state)

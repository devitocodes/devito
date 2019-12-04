from functools import partial

from cached_property import cached_property

from devito.ir import FindNodes
from devito.targets import (CustomRewriter, Intel64Rewriter, Ompizer, avoid_denormals,
                            loop_wrapping, parallelize_shm, insert_defs, insert_casts)

from devito.yask.utils import Offloaded

__all__ = ['YaskRewriter', 'YaskCustomRewriter']


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


class YaskCustomRewriter(CustomRewriter, YaskRewriter):

    @cached_property
    def passes_mapper(self):
        return {
            'denormals': partial(avoid_denormals),
            'wrapping': partial(loop_wrapping),
            'openmp': partial(parallelize_shm, parallelizer_shm=self.parallelizer_shm)
        }

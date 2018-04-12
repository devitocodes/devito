from devito.dle.backends import AdvancedRewriter, Ompizer, dle_pass

__all__ = ['YaskRewriter']


class YaskOmpizer(Ompizer):
    # TODO: will need to specialize `_make_omp_parallel_tree` as soon as the
    # necessary APIs will be ready in YASK (e.g., atomic incs required)

    def key(self, v):
        return v.is_Parallel and not (v.is_Elementizable or v.is_Vectorizable)


class YaskRewriter(AdvancedRewriter):

    _parallelizer = YaskOmpizer

    def _pipeline(self, state):
        self._avoid_denormals(state)
        if self.params['openmp'] is True:
            self._parallelize(state)

    @dle_pass
    def _parallelize(self, iet, state):
        def key(i):
            # TODO: ParallelRelaxed not supported yet (see TODO above)
            return i.is_Parallel and not (i.is_Elementizable or i.is_Vectorizable)
        return self._parallelizer(key).make_omp_parallel_iet(iet), {}

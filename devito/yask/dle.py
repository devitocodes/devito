from devito.dle import AdvancedRewriter, Ompizer

__all__ = ['YaskRewriter']


class YaskOmpizer(Ompizer):

    pass


class YaskRewriter(AdvancedRewriter):

    _shm_parallelizer_type = YaskOmpizer

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._loop_wrapping(state)
        if self.params['openmp'] is True:
            self._shm_parallelize(state)

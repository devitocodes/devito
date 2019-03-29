from devito.dle import Intel64Rewriter

__all__ = ['YaskRewriter']


class YaskRewriter(Intel64Rewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._loop_wrapping(state)
        if self.params['openmp'] is True:
            self._node_parallelize(state)

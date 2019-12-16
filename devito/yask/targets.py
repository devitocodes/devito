from functools import partial

from cached_property import cached_property

from devito.ir import FindNodes
from devito.targets import (CustomTarget, CPU64NoopTarget, DataManager, Ompizer,
                            avoid_denormals, loop_wrapping)

from devito.yask.utils import Offloaded

__all__ = ['YaskTarget', 'YaskCustomTarget']


class YaskOmpizer(Ompizer):

    def __init__(self, key=None):
        if key is None:
            def key(i):
                # If it's not parallel, nothing to do
                if not i.is_ParallelRelaxed or i.is_Vectorized:
                    return False
                # If some of the inner computation has been offloaded to YASK,
                # avoid introducing an outer level of parallelism
                if FindNodes(Offloaded).visit(i):
                    return False
                return True
        super(YaskOmpizer, self).__init__(key=key)


class YaskTarget(CPU64NoopTarget):

    def _pipeline(self, graph):
        # Shared-memory and SIMD-level parallelism
        if self.params['openmp']:
            YaskOmpizer().make_parallel(graph)

        # Misc optimizations
        avoid_denormals(graph)
        loop_wrapping(graph)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)


class YaskCustomTarget(CustomTarget, YaskTarget):

    @cached_property
    def passes_mapper(self):
        ompizer = YaskOmpizer()

        return {
            'denormals': partial(avoid_denormals),
            'wrapping': partial(loop_wrapping),
            'openmp': partial(ompizer.make_parallel),
        }

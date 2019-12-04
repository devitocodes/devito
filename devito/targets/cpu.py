from functools import partial

from cached_property import cached_property

from devito.exceptions import DLEException
from devito.targets.basic import PlatformRewriter
from devito.targets.common import (Blocker, Ompizer, avoid_denormals, insert_defs,
                                   insert_casts, optimize_halospots, parallelize_dist,
                                   loop_blocking, loop_wrapping, simdize, parallelize_shm,
                                   minimize_remainders, hoist_prodders)

__all__ = ['CPU64NoopRewriter', 'CPU64Rewriter', 'Intel64Rewriter', 'PowerRewriter',
           'ArmRewriter', 'CustomRewriter']
#TODO: change all these Rewriter names


class CPU64NoopRewriter(PlatformRewriter):

    def _pipeline(self, graph):
        # Symbol definitions
        insert_defs(graph)
        insert_casts(graph)


class CPU64Rewriter(CPU64NoopRewriter):

    #TODO: move this
    _default_blocking_levels = 1
    """
    Depth of the loop blocking hierarchy. 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    def __init__(self, params, platform):
        super(CPU64Rewriter, self).__init__(params, platform)

        # Iteration blocker (i.e., for "loop blocking")
        self.blocker = Blocker(params.get('blockinner'),
                               params.get('blocklevels') or self._default_blocking_levels)

        # Shared-memory parallelizer
        self.parallelizer_shm = Ompizer()

    def _pipeline(self, graph):
        # Optimization and parallelism
        avoid_denormals(graph)
        optimize_halospots(graph)
        if self.params['mpi']:
            parallelize_dist(graph, mode=self.params['mpi'])
        loop_blocking(graph, blocker=self.blocker)
        simdize(graph, simd_reg_size=self.platform.simd_reg_size)
        if self.params['openmp']:
            parallelize_shm(graph, parallelizer_shm=self.parallelizer_shm)
        minimize_remainders(graph, simd_items_per_reg=self.platform.simd_items_per_reg)
        hoist_prodders(graph)

        # Symbol definitions
        insert_defs(graph)
        insert_casts(graph)


Intel64Rewriter = CPU64Rewriter
PowerRewriter = CPU64Rewriter
ArmRewriter = CPU64Rewriter


#TODO : the stuff below needs adding iet_insert_decls, iet_insert_casts, etc


class CustomRewriter(CPU64Rewriter):

    def __init__(self, passes, params, platform):
        super(CustomRewriter, self).__init__(params, platform)

        try:
            passes = passes.split(',')
            if 'openmp' not in passes and params['openmp']:
                passes.append('openmp')
        except AttributeError:
            # Already in tuple format
            if not all(i in self.passes_mapper for i in passes):
                raise DLEException("Unknown passes `%s`" % str(passes))
        self.passes = passes

    @cached_property
    def passes_mapper(self):
        return {
            'denormals': partial(avoid_denormals),
            'optcomms': partial(optimize_halospots),
            'wrapping': partial(loop_wrapping),
            'blocking': partial(loop_blocking, blocker=self.blocker),
            'openmp': partial(parallelize_shm, parallelizer_shm=self.parallelizer_shm),
            'mpi': partial(parallelize_dist, mode=self.params['mpi']),
            'simd': partial(simdize, simd_reg_size=self.platform.simd_reg_size),
            'minrem': partial(minimize_remainders,
                              simd_items_per_reg=self.platform.simd_items_per_reg),
            'prodders': partial(hoist_prodders)
        }

    def _pipeline(self, graph):
        for i in self.passes:
            self.passes_mapper[i](graph)

        # Symbol definitions
        insert_defs(graph)
        insert_casts(graph)

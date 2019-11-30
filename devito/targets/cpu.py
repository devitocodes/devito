from functools import partial

from cached_property import cached_property

from devito.exceptions import DLEException
from devito.targets.basic import PlatformRewriter
from devito.targets.common import (Ompizer, avoid_denormals,
                                   optimize_halospots, parallelize_dist, loop_blocking,
                                   loop_wrapping, simdize, parallelize_shm,
                                   minimize_remainders, hoist_prodders)

__all__ = ['CPU64Rewriter', 'Intel64Rewriter', 'PowerRewriter', 'ArmRewriter',
           'CustomRewriter']


class CPU64Rewriter(PlatformRewriter):

    _parallelizer_shm_type = Ompizer

    def _pipeline(self, state):
        # Optimization and parallelism
        avoid_denormals(state)
        optimize_halospots(state)
        if self.params['mpi']:
            parallelize_dist(state, mode=self.params['mpi'])
        loop_blocking(state, blocker=self.blocker)
        simdize(state, simd_reg_size=self.platform.simd_reg_size)
        if self.params['openmp']:
            parallelize_shm(state, parallelizer_shm=self.parallelizer_shm)
        minimize_remainders(state, simd_items_per_reg=self.platform.simd_items_per_reg)
        hoist_prodders(state)


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

    def _pipeline(self, state):
        for i in self.passes:
            self.passes_mapper[i](state)

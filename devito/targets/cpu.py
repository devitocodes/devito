from functools import partial

from cached_property import cached_property

from devito.exceptions import DLEException
from devito.targets.basic import PlatformRewriter
from devito.targets.common import (Ompizer, _avoid_denormals,
                                   _optimize_halospots, _parallelize_dist, _loop_blocking,
                                   _loop_wrapping, _simdize, _parallelize_shm,
                                   _minimize_remainders, _hoist_prodders)

__all__ = ['CPU64Rewriter', 'Intel64Rewriter', 'PowerRewriter', 'ArmRewriter',
           'SpeculativeRewriter', 'CustomRewriter']


class CPU64Rewriter(PlatformRewriter):

    _parallelizer_shm_type = Ompizer

    def _pipeline(self, state):
        # Optimization and parallelism
        _avoid_denormals(state)
        _optimize_halospots(state)
        if self.params['mpi']:
            _parallelize_dist(state, mode=self.params['mpi'])
        _loop_blocking(state, blocker=self.blocker)
        _simdize(state, simd_reg_size=self.platform.simd_reg_size)
        if self.params['openmp']:
            _parallelize_shm(state, parallelizer_shm=self.parallelizer_shm)
        _minimize_remainders(state, simd_items_per_reg=self.platform.simd_items_per_reg)
        _hoist_prodders(state)


Intel64Rewriter = CPU64Rewriter
PowerRewriter = CPU64Rewriter
ArmRewriter = CPU64Rewriter


#TODO : the stuff below needs adding iet_insert_decls, iet_insert_casts, etc


class SpeculativeRewriter(CPU64Rewriter):

    def _pipeline(self, state):
        # Optimization and parallelism
        _avoid_denormals(state)
        _optimize_halospots(state)
        _loop_wrapping(state)
        if self.params['mpi']:
            _parallelize_dist(state, mode=self.params['mpi'])
        _loop_blocking(state, blocker=self.blocker)
        _simdize(state, simd_reg_size=self.platform.simd_reg_size)
        if self.params['openmp']:
            _parallelize_shm(state, parallelizer_shm=self.parallelizer_shm)
        _minimize_remainders(state, simd_items_per_reg=self.platform.simd_items_per_reg)
        _hoist_prodders(state)


class CustomRewriter(SpeculativeRewriter):

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
            'denormals': partial(_avoid_denormals),
            'optcomms': partial(_optimize_halospots),
            'wrapping': partial(_loop_wrapping),
            'blocking': partial(_loop_blocking, blocker=self.blocker),
            'openmp': partial(_parallelize_shm, parallelizer_shm=self.parallelizer_shm),
            'mpi': partial(_parallelize_dist, mode=self.params['mpi']),
            'simd': partial(_simdize, simd_reg_size=self.platform.simd_reg_size),
            'minrem': partial(_minimize_remainders,
                              simd_items_per_reg=self.platform.simd_items_per_reg),
            'prodders': partial(_hoist_prodders)
        }

    def _pipeline(self, state):
        for i in self.passes:
            self.passes_mapper[i](state)

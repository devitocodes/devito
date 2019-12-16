from functools import partial

from cached_property import cached_property

from devito.exceptions import DLEException
from devito.targets.basic import Target
from devito.targets.common import (DataManager, Blocker, Ompizer, avoid_denormals,
                                   optimize_halospots, mpiize, loop_wrapping,
                                   minimize_remainders, hoist_prodders)

__all__ = ['CPU64NoopTarget', 'CPU64Target', 'Intel64Target', 'PowerTarget',
           'ArmTarget', 'CustomTarget']


class CPU64NoopTarget(Target):

    def _pipeline(self, graph):
        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)


class CPU64Target(CPU64NoopTarget):

    BLOCK_LEVELS = 1
    """
    Loop blocking depth. So, 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    def _pipeline(self, graph):
        # Distributed-memory parallelism
        optimize_halospots(graph)
        if self.params['mpi']:
            mpiize(graph, mode=self.params['mpi'])

        # Tiling
        blocker = Blocker(self.params['blockinner'],
                          self.params['blocklevels'] or self.BLOCK_LEVELS)
        blocker.make_blocking(graph)

        # Shared-memory and SIMD-level parallelism
        ompizer = Ompizer()
        ompizer.make_simd(graph, simd_reg_size=self.platform.simd_reg_size)
        if self.params['openmp']:
            ompizer.make_parallel(graph)

        # Misc optimizations
        avoid_denormals(graph)
        minimize_remainders(graph, simd_items_per_reg=self.platform.simd_items_per_reg)
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)


Intel64Target = CPU64Target
PowerTarget = CPU64Target
ArmTarget = CPU64Target


class CustomTarget(CPU64Target):

    def __init__(self, passes, params, platform):
        super(CustomTarget, self).__init__(params, platform)

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
        blocker = Blocker(self.params['blockinner'],
                          self.params['blocklevels'] or self.BLOCK_LEVELS)
        ompizer = Ompizer()

        return {
            'denormals': partial(avoid_denormals),
            'optcomms': partial(optimize_halospots),
            'wrapping': partial(loop_wrapping),
            'blocking': partial(blocker.make_blocking),
            'openmp': partial(ompizer.make_parallel),
            'mpi': partial(mpiize, mode=self.params['mpi']),
            'simd': partial(ompizer.make_simd, simd_reg_size=self.platform.simd_reg_size),
            'minrem': partial(minimize_remainders,
                              simd_items_per_reg=self.platform.simd_items_per_reg),
            'prodders': partial(hoist_prodders)
        }

    def _pipeline(self, graph):
        for i in self.passes:
            self.passes_mapper[i](graph)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

from functools import partial

from devito.core.operator import OperatorCore
from devito.exceptions import InvalidOperator
from devito.targets import (DataManager, Blocker, Ompizer, avoid_denormals,
                            optimize_halospots, mpiize, loop_wrapping,
                            minimize_remainders, hoist_prodders)
from devito.tools import as_tuple

__all__ = ['CPU64NoopOperator', 'CPU64Operator', 'Intel64Operator', 'PowerOperator',
           'ArmOperator', 'CustomOperator']


class CPU64NoopOperator(OperatorCore):

    @classmethod
    def _specialize_iet(cls, graph, **kwargs):
        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


class CPU64Operator(CPU64NoopOperator):

    BLOCK_LEVELS = 1
    """
    Loop blocking depth. So, 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    @classmethod
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Tiling
        blocker = Blocker(options['blockinner'],
                          options['blocklevels'] or cls.BLOCK_LEVELS)
        blocker.make_blocking(graph)

        # Shared-memory and SIMD-level parallelism
        ompizer = Ompizer()
        ompizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)
        if options['openmp']:
            ompizer.make_parallel(graph)

        # Misc optimizations
        avoid_denormals(graph)
        minimize_remainders(graph, simd_items_per_reg=platform.simd_items_per_reg)
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


Intel64Operator = CPU64Operator
PowerOperator = CPU64Operator
ArmOperator = CPU64Operator


class CustomOperator(CPU64Operator):

    @classmethod
    def _make_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']

        blocker = Blocker(options['blockinner'],
                          options['blocklevels'] or cls.BLOCK_LEVELS)

        ompizer = Ompizer()

        return {
            'denormals': partial(avoid_denormals),
            'optcomms': partial(optimize_halospots),
            'wrapping': partial(loop_wrapping),
            'blocking': partial(blocker.make_blocking),
            'openmp': partial(ompizer.make_parallel),
            'mpi': partial(mpiize, mode=options['mpi']),
            'simd': partial(ompizer.make_simd, simd_reg_size=platform.simd_reg_size),
            'minrem': partial(minimize_remainders,
                              simd_items_per_reg=platform.simd_items_per_reg),
            'prodders': partial(hoist_prodders)
        }

    @classmethod
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                passes_mapper[i](graph)
            except KeyError:
                raise InvalidOperator("Unknown passes `%s`" % str(passes))

        # Force-call `mpi` if requested via global option
        if 'mpi' not in passes and options['mpi']:
            passes_mapper['mpi'](graph)

        # Force-call `openmp` if requested via global option
        if 'openmp' not in passes and options['openmp']:
            passes_mapper['openmp'](graph)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph

from devito.core.cpu import CPU64Operator, CPU64OpenMPOperator
from devito.passes.iet import (mpiize, optimize_halospots, hoist_prodders,
                               relax_incr_dimensions)
from devito.tools import timed_pass

__all__ = ['ArmOperator', 'ArmOpenMPOperator']


class ArmOperator(CPU64Operator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Lower IncrDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph, sregistry=sregistry)

        # SIMD-level parallelism
        ompizer = cls._Parallelizer(sregistry, options)
        ompizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._DataManager(sregistry).process(graph)

        return graph


class ArmOpenMPOperator(CPU64OpenMPOperator):

    # Variable set to 4 to avoid nested parallelism on ThunderX2
    PAR_NESTED = 4

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Lower IncrDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph, sregistry=sregistry)

        # SIMD-level parallelism
        ompizer = cls._Parallelizer(sregistry, options)
        ompizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)

        # Shared-memory parallelism
        ompizer.make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._DataManager(sregistry).process(graph)

        return graph

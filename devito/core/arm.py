from devito.core.cpu import Cpu64AdvOperator, Cpu64AdvOmpOperator
from devito.passes.iet import (mpiize, optimize_halospots, hoist_prodders,
                               relax_incr_dimensions)
from devito.tools import timed_pass

__all__ = ['ArmAdvOperator', 'ArmAdvOmpOperator']


class ArmAdvOperator(Cpu64AdvOperator):

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
        parizer = cls._Parallelizer(sregistry, options)
        parizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._DataManager(cls._Parallelizer, sregistry).process(graph)

        return graph


class ArmAdvOmpOperator(Cpu64AdvOmpOperator):

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
        parizer = cls._Parallelizer(sregistry, options)
        parizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)

        # Shared-memory parallelism
        parizer.make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._DataManager(cls._Parallelizer, sregistry).process(graph)

        # Initialize the target-language runtime
        parizer.initialize(graph)

        return graph

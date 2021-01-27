from devito.core.cpu import Cpu64AdvOperator, Cpu64COperatorMixin, Cpu64OmpOperatorMixin
from devito.passes.iet import (mpiize, optimize_halospots, hoist_prodders,
                               relax_incr_dimensions)
from devito.tools import timed_pass

__all__ = ['ArmAdvCOperator', 'ArmAdvOmpOperator']


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

        # Parallelism
        parizer = cls._Parallelizer(sregistry, options, platform)
        parizer.make_simd(graph)
        parizer.make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._DataManager(cls._Parallelizer, sregistry).process(graph)

        # Initialize the target-language runtime
        parizer.initialize(graph)

        return graph


class ArmAdvCOperator(Cpu64COperatorMixin, ArmAdvOperator):
    pass


class ArmAdvOmpOperator(Cpu64OmpOperatorMixin, ArmAdvOperator):
    PAR_NESTED = 4  # Avoid nested parallelism on ThunderX2

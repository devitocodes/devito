from devito.ir.iet import Iteration, FindNodes, Transformer, VECTOR
from devito.targets.basic import PlatformRewriter
from devito.targets.common import (OmpizerGPU, dle_pass, insert_defs, insert_casts,
                                   optimize_halospots, parallelize_dist,
                                   parallelize_shm, hoist_prodders)

__all__ = ['DeviceOffloadingRewriter']


@dle_pass
def simdize(iet):
    # No SIMD-ization for devices. We then drop the VECTOR property
    # so that later passes can perform more aggressive transformations
    mapper = {}
    for i in FindNodes(Iteration).visit(iet):
        if i.is_Vectorizable:
            properties = [p for p in i.properties if p is not VECTOR]
            mapper[i] = i._rebuild(properties=properties)

    iet = Transformer(mapper).visit(iet)

    return iet, {}


#TODO: Move here OmpizerGPU


class DeviceOffloadingRewriter(PlatformRewriter):

    def __init__(self, params, platform):
        super(DeviceOffloadingRewriter, self).__init__(params, platform)

        # Shared-memory parallelizer
        self.parallelizer_shm = OmpizerGPU()

    def _pipeline(self, graph):
        # Optimization and parallelism
        optimize_halospots(graph)
        if self.params['mpi']:
            parallelize_dist(graph, mode=self.params['mpi'])
        simdize(graph)
        if self.params['openmp']:
            parallelize_shm(graph, parallelizer_shm=self.parallelizer_shm)
        hoist_prodders(graph)

        # Symbol definitions
        #TODO
        #insert_defs(graph)
        #insert_casts(graph)

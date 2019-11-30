from devito.ir.iet import Iteration, FindNodes, Transformer, VECTOR
from devito.targets.basic import PlatformRewriter
from devito.targets.common import (OmpizerGPU, dle_pass,
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

    _parallelizer_shm_type = OmpizerGPU

    def _pipeline(self, state):
        optimize_halospots(state)
        if self.params['mpi']:
            parallelize_dist(state, mode=self.params['mpi'])
        simdize(state)
        if self.params['openmp']:
            parallelize_shm(state, parallelizer_shm=self.parallelizer_shm)
        hoist_prodders(state)

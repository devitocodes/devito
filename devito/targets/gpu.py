from devito.ir.iet import Iteration, FindNodes, Transformer, VECTOR
from devito.targets.basic import PlatformRewriter
from devito.targets.common import (OmpizerGPU, dle_pass,
                                   _optimize_halospots, _parallelize_dist,
                                   _parallelize_shm, _hoist_prodders)

__all__ = ['DeviceOffloadingRewriter']


@dle_pass
def _simdize(iet):
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
        _optimize_halospots(state)
        if self.params['mpi']:
            _parallelize_dist(state, mode=self.params['mpi'])
        _simdize(state)
        if self.params['openmp']:
            _parallelize_shm(state, parallelizer_shm=self.parallelizer_shm)
        _hoist_prodders(state)

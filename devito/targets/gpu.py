import cgen as c

from devito.data import FULL
from devito.ir.iet import (Expression, Iteration, List, FindNodes, MapNodes,
                           Transformer, COLLAPSED, VECTOR, retrieve_iteration_tree)
from devito.targets.basic import Target
from devito.targets.common import (DataManager, Ompizer, ParallelTree, target_pass,
                                   optimize_halospots, mpiize, hoist_prodders)
from devito.tools import filter_sorted, flatten

__all__ = ['DeviceOffloadingTarget']


class OffloadingOmpizer(Ompizer):

    COLLAPSE_NCORES = 1
    """
    Always collapse when possible.
    """

    COLLAPSE_WORK = 1
    """
    Always collapse when possible.
    """

    lang = dict(Ompizer.lang)
    lang.update({
        'par-for-teams': lambda i:
            c.Pragma('omp target teams distribute parallel for collapse(%d)' % i),
        'map-enter': lambda i, j:
            c.Pragma('omp target enter data map(to: %s%s)' % (i, j)),
        'map-exit': lambda i, j:
            c.Pragma('omp target exit data map(from: %s%s)' % (i, j))
    })

    def __init__(self, key=None):
        if key is None:
            key = lambda i: i.is_ParallelRelaxed
        super(OffloadingOmpizer, self).__init__(key=key)

    def __map_data(self, f):
        if f.is_Array:
            return f.symbolic_shape
        else:
            return tuple(f._C_get_field(FULL, d).size for d in f.dimensions)

    def __map_to(self, f):
        return self.lang['map-enter'](f.name,
                                      ''.join('[0:%s]' % i for i in self.__map_data(f)))

    def __map_from(self, f):
        return self.lang['map-exit'](f.name,
                                     ''.join('[0:%s]' % i for i in self.__map_data(f)))

    def _make_threaded_prodders(self, partree):
        # no-op for now
        return partree

    def _make_partree(self, candidates, nthreads=None):
        """
        Parallelize the `candidates` Iterations attaching suitable OpenMP pragmas
        for GPU offloading.
        """
        assert candidates
        root = candidates[0]

        # Get the collapsable Iterations
        collapsable = self._find_collapsable(root, candidates)
        ncollapse = 1 + len(collapsable)

        # Prepare to build a ParallelTree
        omp_pragma = self.lang['par-for-teams'](ncollapse)

        # Create a ParallelTree
        body = root._rebuild(pragmas=root.pragmas + (omp_pragma,),
                             properties=root.properties + (COLLAPSED(ncollapse),))
        partree = ParallelTree([], body, nthreads=nthreads)

        collapsed = [partree] + collapsable

        return root, partree, collapsed

    def _make_parregion(self, partree):
        # no-op for now
        return partree

    def _make_guard(self, partree, *args):
        # no-op for now
        return partree

    def _make_nested_partree(self, partree):
        # no-op for now
        return partree

    def _make_data_transfers(self, iet):
        data_transfers = {}
        for iteration, partree in MapNodes(child_types=ParallelTree).visit(iet).items():
            # The data that need to be transfered between host and device
            exprs = FindNodes(Expression).visit(partree)
            reads = set().union(*[i.reads for i in exprs])
            writes = set([i.write for i in exprs])
            data = {i for i in reads | writes if i.is_Tensor}

            # At what depth in the IET should the data transfer be performed?
            candidate = iteration
            for tree in retrieve_iteration_tree(iet):
                if iteration not in tree:
                    continue
                for i in reversed(tree[:tree.index(iteration)+1]):
                    found = False
                    for k, v in MapNodes('any', Expression, 'groupby').visit(i).items():
                        test0 = any(isinstance(n, ParallelTree) for n in k)
                        test1 = set().union(*[e.functions for e in v]) & data
                        if not test0 and test1:
                            found = True
                            break
                    if found:
                        break
                    candidate = i
                break

            # Create the omp pragmas for the data transfer
            map_tos = [self.__map_to(i) for i in data]
            map_froms = [self.__map_from(i) for i in writes if i.is_Tensor]
            data_transfers.setdefault(candidate, []).append((map_tos, map_froms))

        # Now create a new IET with the data transfer
        mapper = {}
        for i, v in data_transfers.items():
            map_tos, map_froms = zip(*v)
            map_tos = filter_sorted(flatten(map_tos), key=lambda i: i.value)
            map_froms = filter_sorted(flatten(map_froms), key=lambda i: i.value)
            mapper[i] = List(header=map_tos, body=i, footer=map_froms)
        iet = Transformer(mapper).visit(iet)

        return iet

    @target_pass
    def make_simd(self, iet, **kwargs):
        # No SIMD-ization for devices. We then drop the VECTOR property
        # so that later passes can perform more aggressive transformations
        mapper = {}
        for i in FindNodes(Iteration).visit(iet):
            if i.is_Vectorizable:
                properties = [p for p in i.properties if p is not VECTOR]
                mapper[i] = i._rebuild(properties=properties)

        iet = Transformer(mapper).visit(iet)

        return iet, {}


class OffloadingDataManager(DataManager):
    pass


class DeviceOffloadingTarget(Target):

    def __init__(self, params, platform):
        super(DeviceOffloadingTarget, self).__init__(params, platform)

        # Shared-memory parallelizer
        self.ompizer = OffloadingOmpizer()

        # Data manager (declarations, definitions, movemented between
        # host and device, ...)
        self.data_manager = OffloadingDataManager()

    def _pipeline(self, graph):
        # Optimization and parallelism
        optimize_halospots(graph)
        if self.params['mpi']:
            mpiize(graph, mode=self.params['mpi'])
        self.ompizer.make_simd(graph, simd_reg_size=self.platform.simd_reg_size)
        if self.params['openmp']:
            self.ompizer.make_parallel(graph)
        hoist_prodders(graph)

        # Symbol definitions
        #self.data_manager.place_definitions(graph)
        #self.data_manager.place_casts(graph)

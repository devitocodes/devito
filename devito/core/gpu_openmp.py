from functools import partial

import cgen as c

from devito.core.operator import OperatorCore
from devito.data import FULL
from devito.exceptions import InvalidOperator
from devito.ir.clusters import Toposort
from devito.ir.support import COLLAPSED
from devito.passes.clusters import Lift, fuse, scalarize, eliminate_arrays, rewrite
from devito.passes.iet import (DataManager, Ompizer, ParallelTree, optimize_halospots,
                               mpiize, hoist_prodders)
from devito.tools import as_tuple, generator, timed_pass

__all__ = ['DeviceOpenMPNoopOperator', 'DeviceOpenMPOperator',
           'DeviceOpenMPCustomOperator']


class DeviceOmpizer(Ompizer):

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
        'map-enter-to': lambda i, j:
            c.Pragma('omp target enter data map(to: %s%s)' % (i, j)),
        'map-enter-alloc': lambda i, j:
            c.Pragma('omp target enter data map(alloc: %s%s)' % (i, j)),
        'map-exit-from': lambda i, j:
            c.Pragma('omp target exit data map(from: %s%s)' % (i, j)),
        'map-exit-delete': lambda i, j:
            c.Pragma('omp target exit data map(delete: %s%s)' % (i, j)),
    })

    def __init__(self, key=None):
        if key is None:
            key = lambda i: i.is_ParallelRelaxed
        super(DeviceOmpizer, self).__init__(key=key)

    @classmethod
    def _map_data(cls, f):
        if f.is_Array:
            return f.symbolic_shape
        else:
            return tuple(f._C_get_field(FULL, d).size for d in f.dimensions)

    @classmethod
    def _map_to(cls, f):
        return cls.lang['map-enter-to'](f.name, ''.join('[0:%s]' % i
                                                        for i in cls._map_data(f)))

    @classmethod
    def _map_alloc(cls, f):
        return cls.lang['map-enter-alloc'](f.name, ''.join('[0:%s]' % i
                                                           for i in cls._map_data(f)))

    @classmethod
    def _map_from(cls, f):
        return cls.lang['map-exit-from'](f.name, ''.join('[0:%s]' % i
                                                         for i in cls._map_data(f)))

    @classmethod
    def _map_delete(cls, f):
        return cls.lang['map-exit-delete'](f.name, ''.join('[0:%s]' % i
                                                           for i in cls._map_data(f)))

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


class DeviceDataManager(DataManager):

    def _alloc_array_on_high_bw_mem(self, obj, storage):
        if obj in storage._high_bw_mem:
            return

        decl = "(*%s)%s" % (obj.name, "".join("[%s]" % i for i in obj.symbolic_shape[1:]))
        decl = c.Value(obj._C_typedata, decl)

        alloc = DeviceOmpizer._map_alloc(obj)

        free = DeviceOmpizer._map_delete(obj)

        storage._high_bw_mem[obj] = (decl, alloc, free)

    def _map_function_on_high_bw_mem(self, obj, storage, read_only=False):
        if obj in storage._high_bw_mem:
            return

        alloc = DeviceOmpizer._map_to(obj)

        if read_only is False:
            free = DeviceOmpizer._map_from(obj)
        else:
            free = DeviceOmpizer._map_delete(obj)

        storage._high_bw_mem[obj] = (None, alloc, free)


class DeviceOpenMPNoopOperator(OperatorCore):

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        # TODO: this is currently identical to CPU64NoopOperator._specialize_clusters,
        # but it will have to change

        # To create temporaries
        counter = generator()
        template = lambda: "r%d" % counter()

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = Toposort().process(clusters)
        clusters = fuse(clusters)

        # Flop reduction via the DSE
        clusters = rewrite(clusters, template, **kwargs)

        # Lifting
        clusters = Lift().process(clusters)

        # Lifting may create fusion opportunities, which in turn may enable
        # further optimizations
        clusters = fuse(clusters)
        clusters = eliminate_arrays(clusters, template)
        clusters = scalarize(clusters, template)

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']

        # Distributed-memory parallelism
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Shared-memory parallelism
        assert options['openmp']
        DeviceOmpizer().make_parallel(graph)

        # Symbol definitions
        data_manager = DeviceDataManager()
        data_manager.place_definitions(graph, efuncs=list(graph.efuncs.values()))
        data_manager.place_casts(graph)

        return graph


class DeviceOpenMPOperator(DeviceOpenMPNoopOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Shared-memory parallelism
        assert options['openmp']
        DeviceOmpizer().make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DeviceDataManager()
        data_manager.place_definitions(graph, efuncs=list(graph.efuncs.values()))
        data_manager.place_casts(graph)

        return graph


class DeviceOpenMPCustomOperator(DeviceOpenMPOperator):

    @classmethod
    def _make_passes_mapper(cls, **kwargs):
        options = kwargs['options']

        ompizer = DeviceOmpizer()

        return {
            'optcomms': partial(optimize_halospots),
            'openmp': partial(ompizer.make_parallel),
            'mpi': partial(mpiize, mode=options['mpi']),
            'prodders': partial(hoist_prodders)
        }

    @classmethod
    @timed_pass(name='specializing.IET')
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

        # `openmp` must have been enabled
        if 'openmp' not in passes:
            assert options['openmp']
            passes_mapper['openmp'](graph)

        # Symbol definitions
        data_manager = DeviceDataManager()
        data_manager.place_definitions(graph, efuncs=list(graph.efuncs.values()))
        data_manager.place_casts(graph)

        return graph

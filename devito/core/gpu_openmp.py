from functools import partial

import cgen as c

from devito.core.operator import OperatorCore
from devito.data import FULL
from devito.exceptions import InvalidOperator
from devito.ir.clusters import Toposort
from devito.ir.iet import MapExprStmts
from devito.logger import warning
from devito.passes.clusters import (Lift, cire, cse, eliminate_arrays, extract_increments,
                                    factorize, freeze, fuse, optimize_pows, scalarize)
from devito.passes.iet import (DataManager, Storage, Ompizer, ParallelIteration,
                               ParallelTree, optimize_halospots, mpiize, hoist_prodders,
                               iet_pass)
from devito.tools import as_tuple, filter_sorted, generator, timed_pass

__all__ = ['DeviceOpenMPNoopOperator', 'DeviceOpenMPOperator',
           'DeviceOpenMPCustomOperator']


class DeviceOpenMPIteration(ParallelIteration):

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'omp target teams distribute parallel for'

    @classmethod
    def _make_clauses(cls, **kwargs):
        kwargs['chunk_size'] = False
        return super(DeviceOpenMPIteration, cls)._make_clauses(**kwargs)


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
        'map-enter-to': lambda i, j:
            c.Pragma('omp target enter data map(to: %s%s)' % (i, j)),
        'map-enter-alloc': lambda i, j:
            c.Pragma('omp target enter data map(alloc: %s%s)' % (i, j)),
        'map-update': lambda i, j:
            c.Pragma('omp target update from(%s%s)' % (i, j)),
        'map-release': lambda i, j:
            c.Pragma('omp target exit data map(release: %s%s)' % (i, j)),
        'map-exit-delete': lambda i, j:
            c.Pragma('omp target exit data map(delete: %s%s)' % (i, j)),
    })

    _Iteration = DeviceOpenMPIteration

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
    def _map_update(cls, f):
        return cls.lang['map-update'](f.name, ''.join('[0:%s]' % i
                                                      for i in cls._map_data(f)))

    @classmethod
    def _map_release(cls, f):
        return cls.lang['map-release'](f.name, ''.join('[0:%s]' % i
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
        # Create a ParallelTree
        body = self._Iteration(ncollapse=ncollapse, **root.args)
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


class DeviceOpenMPDataManager(DataManager):

    _Parallelizer = DeviceOmpizer

    def _alloc_array_on_high_bw_mem(self, obj, storage):
        if obj in storage._high_bw_mem:
            return

        super(DeviceOpenMPDataManager, self)._alloc_array_on_high_bw_mem(obj, storage)

        decl, alloc, free = storage._high_bw_mem[obj]

        alloc = c.Collection([alloc, self._Parallelizer._map_alloc(obj)])
        free = c.Collection([self._Parallelizer._map_delete(obj), free])

        storage._high_bw_mem[obj] = (decl, alloc, free)

    def _map_function_on_high_bw_mem(self, obj, storage, read_only=False):
        """Place a Function in the high bandwidth memory."""
        if obj in storage._high_bw_mem:
            return

        alloc = self._Parallelizer._map_to(obj)

        if read_only is False:
            free = c.Collection([self._Parallelizer._map_update(obj),
                                 self._Parallelizer._map_release(obj)])
        else:
            free = self._Parallelizer._map_delete(obj)

        storage._high_bw_mem[obj] = (None, alloc, free)

    @iet_pass
    def place_ondevice(self, iet, **kwargs):
        efuncs = kwargs['efuncs']

        storage = Storage()

        if iet.is_ElementalFunction:
            return iet, {}

        # Collect written and read-only symbols
        writes = set()
        reads = set()
        for efunc in efuncs:
            for i, v in MapExprStmts().visit(efunc).items():
                if not i.is_Expression:
                    # No-op
                    continue
                if not any(isinstance(j, self._Parallelizer._Iteration) for j in v):
                    # Not an offloaded Iteration tree
                    continue
                if i.write.is_DiscreteFunction:
                    writes.add(i.write)
                reads = (reads | {r for r in i.reads if r.is_DiscreteFunction}) - writes

        # Update `storage`
        for i in filter_sorted(writes):
            self._map_function_on_high_bw_mem(i, storage)
        for i in filter_sorted(reads):
            self._map_function_on_high_bw_mem(i, storage, read_only=True)

        iet = self._dump_storage(iet, storage)

        return iet, {}


class DeviceOpenMPNoopOperator(OperatorCore):

    CIRE_REPEATS_INV = 2
    """
    Number of CIRE passes to detect and optimize away Dimension-invariant expressions.
    """

    CIRE_REPEATS_SOPS = 2
    """
    Number of CIRE passes to detect and optimize away redundant sum-of-products.
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        options = kwargs['options']

        # Strictly unneccesary, but make it clear that this Operator *will*
        # generate OpenMP code, bypassing any `openmp=False` provided in
        # input to Operator
        options.pop('openmp')

        options['cire-repeats'] = {
            'invariants': options.pop('cire-repeats-inv') or cls.CIRE_REPEATS_INV,
            'sops': options.pop('cire-repeats-sops') or cls.CIRE_REPEATS_SOPS
        }

        return kwargs

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']

        # To create temporaries
        counter = generator()
        template = lambda: "r%d" % counter()

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = Toposort().process(clusters)
        clusters = fuse(clusters)

        # Hoist and optimize Dimension-invariant sub-expressions
        clusters = cire(clusters, template, 'invariants', options, platform)
        clusters = Lift().process(clusters)

        # Reduce flops (potential arithmetic alterations)
        clusters = extract_increments(clusters, template)
        clusters = cire(clusters, template, 'sops', options, platform)
        clusters = factorize(clusters)
        clusters = optimize_pows(clusters)
        clusters = freeze(clusters)

        # Reduce flops (no arithmetic alterations)
        clusters = cse(clusters, template)

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

        # GPU parallelism via OpenMP offloading
        DeviceOmpizer().make_parallel(graph)

        # Symbol definitions
        data_manager = DeviceOpenMPDataManager()
        data_manager.place_ondevice(graph, efuncs=list(graph.efuncs.values()))
        data_manager.place_definitions(graph)
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

        # GPU parallelism via OpenMP offloading
        DeviceOmpizer().make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DeviceOpenMPDataManager()
        data_manager.place_ondevice(graph, efuncs=list(graph.efuncs.values()))
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


class DeviceOpenMPCustomOperator(DeviceOpenMPOperator):

    _known_passes = ('optcomms', 'openmp', 'mpi', 'prodders')
    _known_passes_disabled = ('blocking', 'denormals', 'wrapping', 'simd')
    assert not (set(_known_passes) & set(_known_passes_disabled))

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
    def _build(cls, expressions, **kwargs):
        # Sanity check
        passes = as_tuple(kwargs['mode'])
        for i in passes:
            if i not in cls._known_passes:
                if i in cls._known_passes_disabled:
                    warning("Got explicit pass `%s`, but it's unsupported on an "
                            "Operator of type `%s`" % (i, str(cls)))
                else:
                    raise InvalidOperator("Unknown pass `%s`" % i)

        return super(DeviceOpenMPCustomOperator, cls)._build(expressions, **kwargs)

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
                pass

        # Force-call `mpi` if requested via global option
        if 'mpi' not in passes and options['mpi']:
            passes_mapper['mpi'](graph)

        # GPU parallelism via OpenMP offloading
        if 'openmp' not in passes:
            passes_mapper['openmp'](graph)

        # Symbol definitions
        data_manager = DeviceOpenMPDataManager()
        data_manager.place_ondevice(graph, efuncs=list(graph.efuncs.values()))
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph

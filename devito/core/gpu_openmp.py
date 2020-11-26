from functools import partial, singledispatch

import cgen as c
from sympy import Function
import numpy as np

from devito.core.cpu import CustomOperator
from devito.core.operator import OperatorCore
from devito.data import FULL
from devito.exceptions import InvalidOperator
from devito.ir.equations import DummyEq
from devito.ir.iet import (Block, Call, Callable, ElementalFunction, EntryFunction,
                           Expression, List, LocalExpression, ThreadFunction,
                           FindNodes, FindSymbols, MapExprStmts, Transformer)
from devito.mpi.distributed import MPICommObject
from devito.mpi.routines import CopyBuffer, HaloUpdate, IrecvCall, IsendCall, SendRecv
from devito.passes.equations import collect_derivatives, buffering
from devito.passes.clusters import (Blocking, Lift, Streaming, Tasker, cire, cse,
                                    eliminate_arrays, extract_increments, factorize,
                                    fuse, optimize_pows)
from devito.passes.iet import (DataManager, Storage, Ompizer, OpenMPIteration,
                               ParallelTree, Orchestrator, optimize_halospots, mpiize,
                               hoist_prodders, iet_pass)
from devito.symbolics import Byref, ccode
from devito.tools import as_tuple, filter_sorted, timed_pass
from devito.types import Symbol

__all__ = ['DeviceOpenMPNoopOperator', 'DeviceOpenMPOperator',
           'DeviceOpenMPCustomOperator']


class DeviceOpenMPIteration(OpenMPIteration):

    @classmethod
    def _make_header(cls, **kwargs):
        header, kwargs = super()._make_header(**kwargs)
        kwargs.pop('gpu_fit', None)

        return header, kwargs

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'omp target teams distribute parallel for'

    @classmethod
    def _make_clauses(cls, **kwargs):
        kwargs['chunk_size'] = False
        return super(DeviceOpenMPIteration, cls)._make_clauses(**kwargs)


class DeviceOmpizer(Ompizer):

    lang = dict(Ompizer.lang)
    lang.update({
        'map-enter-to': lambda i, j:
            c.Pragma('omp target enter data map(to: %s%s)' % (i, j)),
        'map-enter-alloc': lambda i, j:
            c.Pragma('omp target enter data map(alloc: %s%s)' % (i, j)),
        'map-update': lambda i, j:
            c.Pragma('omp target update from(%s%s)' % (i, j)),
        'map-update-host': lambda i, j:
            c.Pragma('omp target update from(%s%s)' % (i, j)),
        'map-update-device': lambda i, j:
            c.Pragma('omp target update to(%s%s)' % (i, j)),
        'map-release': lambda i, j:
            c.Pragma('omp target exit data map(release: %s%s)'
                     % (i, j)),
        'map-exit-delete': lambda i, j, k:
            c.Pragma('omp target exit data map(delete: %s%s)%s'
                     % (i, j, k)),
    })

    _Iteration = DeviceOpenMPIteration

    def __init__(self, sregistry, options, key=None):
        super().__init__(sregistry, options, key=key)
        self.gpu_fit = options['gpu-fit']
        self.par_disabled = options['par-disabled']

    @classmethod
    def _make_sections_from_imask(cls, f, imask):
        datasize = cls._map_data(f)
        if imask is None:
            imask = [FULL]*len(datasize)
        assert len(imask) == len(datasize)
        sections = []
        for i, j in zip(imask, datasize):
            if i is FULL:
                start, size = 0, j
            else:
                try:
                    start, size = i
                except TypeError:
                    start, size = i, 1
                start = ccode(start)
            sections.append('[%s:%s]' % (start, size))
        return ''.join(sections)

    @classmethod
    def _map_data(cls, f):
        if f.is_Array:
            return f.symbolic_shape
        else:
            return tuple(f._C_get_field(FULL, d).size for d in f.dimensions)

    @classmethod
    def _map_to(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-enter-to'](f.name, sections)

    _map_to_wait = _map_to

    @classmethod
    def _map_alloc(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-enter-alloc'](f.name, sections)

    @classmethod
    def _map_present(cls, f, imask=None):
        return

    @classmethod
    def _map_update(cls, f):
        return cls.lang['map-update'](f.name, ''.join('[0:%s]' % i
                                                      for i in cls._map_data(f)))

    @classmethod
    def _map_update_host(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-update-host'](f.name, sections)

    _map_update_wait_host = _map_update_host

    @classmethod
    def _map_update_device(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-update-device'](f.name, sections)

    _map_update_wait_device = _map_update_device

    @classmethod
    def _map_release(cls, f):
        return cls.lang['map-release'](f.name, ''.join('[0:%s]' % i
                                                       for i in cls._map_data(f)))

    @classmethod
    def _map_delete(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        # This ugly condition is to avoid a copy-back when, due to
        # domain decomposition, the local size of a Function is 0, which
        # would cause a crash
        cond = ' if(%s)' % ' && '.join('(%s != 0)' % i for i in cls._map_data(f))
        return cls.lang['map-exit-delete'](f.name, sections, cond)

    @classmethod
    def _map_pointers(cls, f):
        raise NotImplementedError

    def _make_threaded_prodders(self, partree):
        if isinstance(partree.root, DeviceOpenMPIteration):
            # no-op for now
            return partree
        else:
            return super()._make_threaded_prodders(partree)

    def _make_partree(self, candidates, nthreads=None):
        """
        Parallelize the `candidates` Iterations attaching suitable OpenMP pragmas
        for parallelism. In particular:

            * All parallel Iterations not *writing* to a host Function, that
              is a Function `f` such that ``is_on_device(f) == False`, are offloaded
              to the device.
            * The remaining ones, that is those writing to a host Function,
              are parallelized on the host.
        """
        assert candidates
        root = candidates[0]

        if is_on_device(root, self.gpu_fit, only_writes=True):
            # The typical case: all written Functions are device Functions, that is
            # they're mapped in the device memory. Then we offload `root` to the device

            # Get the collapsable Iterations
            collapsable = self._find_collapsable(root, candidates)
            ncollapse = 1 + len(collapsable)

            body = self._Iteration(gpu_fit=self.gpu_fit, ncollapse=ncollapse, **root.args)
            partree = ParallelTree([], body, nthreads=nthreads)
            collapsed = [partree] + collapsable

            return root, partree, collapsed
        elif not self.par_disabled:
            # Resort to host parallelism
            return super()._make_partree(candidates, nthreads)
        else:
            return root, None, None

    def _make_parregion(self, partree, *args):
        if isinstance(partree.root, DeviceOpenMPIteration):
            # no-op for now
            return partree
        else:
            return super()._make_parregion(partree, *args)

    def _make_guard(self, parregion, *args):
        partrees = FindNodes(ParallelTree).visit(parregion)
        if any(isinstance(i.root, DeviceOpenMPIteration) for i in partrees):
            # no-op for now
            return parregion
        else:
            return super()._make_guard(parregion, *args)

    def _make_nested_partree(self, partree):
        if isinstance(partree.root, DeviceOpenMPIteration):
            # no-op for now
            return partree
        else:
            return super()._make_nested_partree(partree)


class DeviceOpenMPOrchestrator(Orchestrator):

    _Parallelizer = DeviceOmpizer


class DeviceOpenMPDataManager(DataManager):

    _Parallelizer = DeviceOmpizer

    def __init__(self, sregistry, options):
        """
        Parameters
        ----------
        sregistry : SymbolRegistry
            The symbol registry, to quickly access the special symbols that may
            appear in the IET (e.g., `sregistry.threadid`, that is the thread
            Dimension, used by the DataManager for parallel memory allocation).
        options : dict
            The optimization options.
            Accepted: ['gpu-fit'].
            * 'gpu-fit': an iterable of `Function`s that are guaranteed to fit
              in the device memory. By default, all `Function`s except saved
              `TimeFunction`'s are assumed to fit in the device memory.
        """
        super().__init__(sregistry)
        self.gpu_fit = options['gpu-fit']

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
        _storage = Storage()
        super()._alloc_array_on_high_bw_mem(site, obj, _storage)

        allocs = _storage[site].allocs + [self._Parallelizer._map_alloc(obj)]
        frees = [self._Parallelizer._map_delete(obj)] + _storage[site].frees
        storage.update(obj, site, allocs=allocs, frees=frees)

    def _map_function_on_high_bw_mem(self, site, obj, storage, read_only=False):
        """
        Place a Function in the high bandwidth memory.
        """
        alloc = self._Parallelizer._map_to(obj)

        if read_only is False:
            free = c.Collection([self._Parallelizer._map_update(obj),
                                 self._Parallelizer._map_release(obj)])
        else:
            free = self._Parallelizer._map_delete(obj)

        storage.update(obj, site, allocs=alloc, frees=free)

    @iet_pass
    def map_onmemspace(self, iet, **kwargs):

        @singledispatch
        def _map_onmemspace(iet):
            return iet

        @_map_onmemspace.register(Callable)
        def _(iet):
            # Collect written and read-only symbols
            writes = set()
            reads = set()
            for i, v in MapExprStmts().visit(iet).items():
                if not i.is_Expression:
                    # No-op
                    continue
                if not any(isinstance(j, self._Parallelizer._Iteration) for j in v):
                    # Not an offloaded Iteration tree
                    continue
                if i.write.is_DiscreteFunction:
                    writes.add(i.write)
                reads.update({r for r in i.reads if r.is_DiscreteFunction})

            # Populate `storage`
            storage = Storage()
            for i in filter_sorted(writes):
                if is_on_device(i, self.gpu_fit):
                    self._map_function_on_high_bw_mem(iet, i, storage)
            for i in filter_sorted(reads - writes):
                if is_on_device(i, self.gpu_fit):
                    self._map_function_on_high_bw_mem(iet, i, storage, read_only=True)

            iet = self._dump_storage(iet, storage)

            return iet

        @_map_onmemspace.register(ElementalFunction)
        @_map_onmemspace.register(ThreadFunction)
        @_map_onmemspace.register(CopyBuffer)
        @_map_onmemspace.register(SendRecv)
        @_map_onmemspace.register(HaloUpdate)
        def _(iet):
            return iet

        iet = _map_onmemspace(iet)

        return iet, {}


@iet_pass
def initialize(iet, **kwargs):
    """
    Initialize the OpenMP environment.
    """

    @singledispatch
    def _initialize(iet):
        return iet

    @_initialize.register(EntryFunction)
    def _(iet):
        comm = None

        for i in iet.parameters:
            if isinstance(i, MPICommObject):
                comm = i
                break

        if comm is not None:
            rank = Symbol(name='rank')
            rank_decl = LocalExpression(DummyEq(rank, 0))
            rank_init = Call('MPI_Comm_rank', [comm, Byref(rank)])

            ngpus = Symbol(name='ngpus')
            call = Function('omp_get_num_devices')()
            ngpus_init = LocalExpression(DummyEq(ngpus, call))

            set_device_num = Call('omp_set_default_device', [rank % ngpus])

            body = [rank_decl, rank_init, ngpus_init, set_device_num]

            init = List(header=c.Comment('Begin of OpenMP+MPI setup'),
                        body=body,
                        footer=(c.Comment('End of OpenMP+MPI setup'), c.Line()))

            iet = iet._rebuild(body=(init,) + iet.body)

        return iet

    iet = _initialize(iet)

    return iet, {}


@iet_pass
def mpi_gpu_direct(iet, **kwargs):
    """
    Modify MPI Callables to enable multiple GPUs performing GPU-Direct communication.
    """
    mapper = {}
    for node in FindNodes((IsendCall, IrecvCall)).visit(iet):
        header = c.Pragma('omp target data use_device_ptr(%s)' %
                          node.arguments[0].name)
        mapper[node] = Block(header=header, body=node)

    iet = Transformer(mapper).visit(iet)

    return iet, {}


class DeviceOpenMPNoopOperator(OperatorCore):

    BLOCK_LEVELS = 1
    """
    Loop blocking depth. So, 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    CIRE_REPEATS_INV = 2
    """
    Number of CIRE passes to detect and optimize away Dimension-invariant expressions.
    """

    CIRE_REPEATS_SOPS = 7
    """
    Number of CIRE passes to detect and optimize away redundant sum-of-products.
    """

    CIRE_MINCOST_INV = 50
    """
    Minimum operation count of a Dimension-invariant aliasing expression to be
    optimized away. Dimension-invariant aliases are lifted outside of one or more
    invariant loop(s), so they require tensor temporaries that can be potentially
    very large (e.g., the whole domain in the case of time-invariant aliases).
    """

    CIRE_MINCOST_SOPS = 10
    """
    Minimum operation count of a sum-of-product aliasing expression to be optimized away.
    """

    PAR_CHUNK_NONAFFINE = 3
    """
    Coefficient to adjust the chunk size in non-affine parallel loops.
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        o = {}
        oo = kwargs['options']

        # Execution modes
        o['mpi'] = oo.pop('mpi')

        # Strictly unneccesary, but make it clear that this Operator *will*
        # generate OpenMP code, bypassing any `openmp=False` provided in
        # input to Operator
        oo.pop('openmp')

        # Buffering
        o['buf-async-degree'] = oo.pop('buf-async-degree', None)

        # Blocking
        o['blockinner'] = oo.pop('blockinner', True)
        o['blocklevels'] = oo.pop('blocklevels', cls.BLOCK_LEVELS)

        # CIRE
        o['min-storage'] = False
        o['cire-rotate'] = False
        o['cire-onstack'] = False
        o['cire-maxpar'] = oo.pop('cire-maxpar', True)
        o['cire-maxalias'] = oo.pop('cire-maxalias', False)
        o['cire-repeats'] = {
            'invariants': oo.pop('cire-repeats-inv', cls.CIRE_REPEATS_INV),
            'sops': oo.pop('cire-repeats-sops', cls.CIRE_REPEATS_SOPS)
        }
        o['cire-mincost'] = {
            'invariants': oo.pop('cire-mincost-inv', cls.CIRE_MINCOST_INV),
            'sops': oo.pop('cire-mincost-sops', cls.CIRE_MINCOST_SOPS)
        }

        # GPU parallelism
        o['par-collapse-ncores'] = 1  # Always use a collapse clause
        o['par-collapse-work'] = 1  # Always use a collapse clause
        o['par-chunk-nonaffine'] = oo.pop('par-chunk-nonaffine', cls.PAR_CHUNK_NONAFFINE)
        o['par-dynamic-work'] = np.inf  # Always use static scheduling
        o['par-nested'] = np.inf  # Never use nested parallelism
        o['par-disabled'] = oo.pop('par-disabled', True)  # No host parallelism by default
        o['gpu-direct'] = oo.pop('gpu-direct', True)
        o['gpu-fit'] = as_tuple(oo.pop('gpu-fit', None))

        if oo:
            raise InvalidOperator("Unsupported optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # GPU parallelism via OpenMP offloading
        DeviceOmpizer(sregistry, options).make_parallel(graph)

        # Symbol definitions
        DeviceOpenMPDataManager(sregistry, options).process(graph)

        # Initialize OpenMP environment
        initialize(graph)

        return graph


class DeviceOpenMPOperator(DeviceOpenMPNoopOperator):

    @classmethod
    @timed_pass(name='specializing.DSL')
    def _specialize_dsl(cls, expressions, **kwargs):
        expressions = collect_derivatives(expressions)

        return expressions

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = fuse(clusters, toposort=True)

        # Hoist and optimize Dimension-invariant sub-expressions
        clusters = cire(clusters, 'invariants', sregistry, options, platform)
        clusters = Lift().process(clusters)

        # Reduce flops (potential arithmetic alterations)
        clusters = extract_increments(clusters, sregistry)
        clusters = cire(clusters, 'sops', sregistry, options, platform)
        clusters = factorize(clusters)
        clusters = optimize_pows(clusters)

        # Reduce flops (no arithmetic alterations)
        clusters = cse(clusters, sregistry)

        # Lifting may create fusion opportunities, which in turn may enable
        # further optimizations
        clusters = fuse(clusters)
        clusters = eliminate_arrays(clusters)

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # GPU parallelism via OpenMP offloading
        DeviceOmpizer(sregistry, options).make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        DeviceOpenMPDataManager(sregistry, options).process(graph)

        # Initialize OpenMP environment
        initialize(graph)
        # TODO: This should be moved right below the `mpiize` pass, but currently calling
        # `mpi_gpu_direct` before Symbol definitions` block would create Blocks before
        # creating C variables. That would lead to MPI_Request variables being local to
        # their blocks. This way, it would generate incorrect C code.
        if options['gpu-direct']:
            mpi_gpu_direct(graph)

        return graph


class DeviceOpenMPCustomOperator(CustomOperator, DeviceOpenMPOperator):

    _normalize_kwargs = DeviceOpenMPOperator._normalize_kwargs

    @classmethod
    def _make_exprs_passes_mapper(cls, **kwargs):
        options = kwargs['options']

        # This callback is used by `buffering` to replace host Functions with
        # Arrays, used as device buffers for streaming-in and -out of data
        def callback(f):
            if not is_on_device(f, options['gpu-fit']):
                return [f.time_dim]
            else:
                return None

        return {
            'buffering': lambda i: buffering(i, callback, options)
        }

    @classmethod
    def _make_clusters_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        runs_on_host, reads_if_on_host = make_callbacks(options)

        return {
            'blocking': Blocking(options).process,
            'tasking': Tasker(runs_on_host).process,
            'streaming': Streaming(reads_if_on_host).process,
            'factorize': factorize,
            'fuse': fuse,
            'lift': lambda i: Lift().process(cire(i, 'invariants', sregistry,
                                                  options, platform)),
            'cire-sops': lambda i: cire(i, 'sops', sregistry, options, platform),
            'cse': lambda i: cse(i, sregistry),
            'opt-pows': optimize_pows,
            'topofuse': lambda i: fuse(i, toposort=True)
        }

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        ompizer = DeviceOmpizer(sregistry, options)
        orchestrator = DeviceOpenMPOrchestrator(sregistry)

        return {
            'optcomms': partial(optimize_halospots),
            'openmp': partial(ompizer.make_parallel),
            'orchestrate': partial(orchestrator.process),
            'mpi': partial(mpiize, mode=options['mpi']),
            'prodders': partial(hoist_prodders),
            'gpu-direct': partial(mpi_gpu_direct)
        }

    _known_passes = (
        # DSL
        'collect-derivs',
        # Expressions
        'buffering',
        # Clusters
        'blocking', 'tasking', 'streaming', 'factorize', 'fuse', 'lift',
        'cire-sops', 'cse', 'opt-pows', 'topofuse',
        # IET
        'optcomms', 'openmp', 'orchestrate', 'mpi', 'prodders', 'gpu-direct'
    )
    _known_passes_disabled = ('denormals', 'simd', 'openacc')
    assert not (set(_known_passes) & set(_known_passes_disabled))

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_iet_passes_mapper(**kwargs)

        # Force-call `mpi` if requested via global option
        if 'mpi' not in passes and options['mpi']:
            passes_mapper['mpi'](graph)

        # GPU parallelism via OpenMP offloading
        if 'openmp' not in passes:
            passes_mapper['openmp'](graph)

        # Call passes
        for i in passes:
            try:
                passes_mapper[i](graph)
            except KeyError:
                pass

        # Symbol definitions
        DeviceOpenMPDataManager(sregistry, options).process(graph)

        # Initialize OpenMP environment
        initialize(graph)

        return graph


# Utils

def is_on_device(maybe_symbol, gpu_fit, only_writes=False):
    """
    True if all given Functions are allocated in the device memory, False otherwise.

    Parameters
    ----------
    maybe_symbol : Indexed or Function or Node
        The inspected object. May be a single Indexed or Function, or even an
        entire piece of IET.
    gpu_fit : list of Function
        The Function's which are known to definitely fit in the device memory. This
        information is given directly by the user through the compiler option
        `gpu-fit` and is propagated down here through the various stages of lowering.
    only_writes : bool, optional
        Only makes sense if `maybe_symbol` is an IET. If True, ignore all Function's
        that do not appear on the LHS of at least one Expression. Defaults to False.
    """
    try:
        functions = (maybe_symbol.function,)
    except AttributeError:
        assert maybe_symbol.is_Node
        iet = maybe_symbol
        functions = set(FindSymbols().visit(iet))
        if only_writes:
            expressions = FindNodes(Expression).visit(iet)
            functions &= {i.write for i in expressions}

    return all(not (f.is_TimeFunction and f.save is not None and f not in gpu_fit)
               for f in functions)


def make_callbacks(options):
    """
    Options-dependent callbacks used by various compiler passes.
    """

    def is_on_host(f):
        return not is_on_device(f, options['gpu-fit'])

    def runs_on_host(c):
        # The only situation in which a Cluster doesn't get offloaded to
        # the device is when it writes to a host Function
        return any(is_on_host(f) for f in c.scope.writes)

    def reads_if_on_host(c):
        if not runs_on_host(c):
            return [f for f in c.scope.reads if is_on_host(f)]
        else:
            return []

    return runs_on_host, reads_if_on_host

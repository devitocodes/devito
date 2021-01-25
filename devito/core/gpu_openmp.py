from functools import singledispatch

import cgen as c
from sympy import Function

from devito.core.gpu import (DeviceNoopOperator, DeviceOperator, DeviceCustomOperator,
                             is_on_device)
from devito.data import FULL
from devito.ir.equations import DummyEq
from devito.ir.iet import (Block, Call, Conditional, EntryFunction,
                           List, LocalExpression, FindNodes,
                           MapExprStmts, Transformer)
from devito.mpi.distributed import MPICommObject
from devito.mpi.routines import IrecvCall, IsendCall
from devito.passes.iet import (DataManager, Ompizer, OpenMPIteration, ParallelTree,
                               Storage, iet_pass)  #TODO: TO BE MOVED...
from devito.symbolics import CondNe, Byref, ccode
from devito.tools import filter_sorted
from devito.types import DeviceID, DeviceRM, Symbol

__all__ = ['DeviceOpenMPNoopOperator', 'DeviceOpenMPOperator',
           'DeviceOpenMPCustomOperator']


# DeviceOpenMP-specific passes


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
        'map-release': lambda i, j, k:
            c.Pragma('omp target exit data map(release: %s%s)%s'
                     % (i, j, k)),
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
    def _map_release(cls, f, devicerm=None):
        return cls.lang['map-release'](f.name,
                                       ''.join('[0:%s]' % i for i in cls._map_data(f)),
                                       (' if(%s)' % devicerm.name) if devicerm else '')

    @classmethod
    def _map_delete(cls, f, imask=None, devicerm=None):
        sections = cls._make_sections_from_imask(f, imask)
        # This ugly condition is to avoid a copy-back when, due to
        # domain decomposition, the local size of a Function is 0, which
        # would cause a crash
        items = []
        if devicerm is not None:
            items.append(devicerm.name)
        items.extend(['(%s != 0)' % i for i in cls._map_data(f)])
        cond = ' if(%s)' % ' && '.join(items)
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

    @iet_pass
    def make_gpudirect(self, iet, **kwargs):
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


class DeviceOpenMPDataManager(DataManager):

    def __init__(self, parallelizer, sregistry, options):
        """
        Parameters
        ----------
        parallelizer : Parallelizer
            Used to implement data movement between host and device as well as
            data mapping, allocation, and deallocation on the device.
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
        super().__init__(parallelizer, sregistry)
        self.gpu_fit = options['gpu-fit']

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
        _storage = Storage()
        super()._alloc_array_on_high_bw_mem(site, obj, _storage)

        allocs = _storage[site].allocs + [self.parallelizer._map_alloc(obj)]
        frees = [self.parallelizer._map_delete(obj)] + _storage[site].frees
        storage.update(obj, site, allocs=allocs, frees=frees)

    def _map_function_on_high_bw_mem(self, site, obj, storage, devicerm, read_only=False):
        """
        Place a Function in the high bandwidth memory.
        """
        alloc = self.parallelizer._map_to(obj)

        if read_only is False:
            free = c.Collection([self.parallelizer._map_update(obj),
                                 self.parallelizer._map_release(obj, devicerm=devicerm)])
        else:
            free = self.parallelizer._map_delete(obj, devicerm=devicerm)

        storage.update(obj, site, allocs=alloc, frees=free)

    @iet_pass
    def map_onmemspace(self, iet, **kwargs):

        @singledispatch
        def _map_onmemspace(iet):
            return iet, {}

        @_map_onmemspace.register(EntryFunction)
        def _(iet):
            # Special symbol which gives user code control over data deallocations
            devicerm = DeviceRM()

            # Collect written and read-only symbols
            writes = set()
            reads = set()
            for i, v in MapExprStmts().visit(iet).items():
                if not i.is_Expression:
                    # No-op
                    continue
                if not any(isinstance(j, self.parallelizer._Iteration) for j in v):
                    # Not an offloaded Iteration tree
                    continue
                if i.write.is_DiscreteFunction:
                    writes.add(i.write)
                reads.update({r for r in i.reads if r.is_DiscreteFunction})

            # Populate `storage`
            storage = Storage()
            for i in filter_sorted(writes):
                if is_on_device(i, self.gpu_fit):
                    self._map_function_on_high_bw_mem(iet, i, storage, devicerm)
            for i in filter_sorted(reads - writes):
                if is_on_device(i, self.gpu_fit):
                    self._map_function_on_high_bw_mem(iet, i, storage, devicerm, True)

            iet = self._dump_storage(iet, storage)

            return iet, {'args': devicerm}

        return _map_onmemspace(iet)


@iet_pass
def initialize(iet, **kwargs):
    """
    Initialize the OpenMP environment.
    """

    @singledispatch
    def _initialize(iet):
        return iet, {}

    @_initialize.register(EntryFunction)
    def _(iet):
        # TODO: we need to pick the rank from `comm_shm`, not `comm`,
        # so that we have nranks == ngpus (as long as the user has launched
        # the right number of MPI processes per node given the available
        # number of GPUs per node)

        objcomm = None
        for i in iet.parameters:
            if isinstance(i, MPICommObject):
                objcomm = i
                break

        deviceid = DeviceID()
        if objcomm is not None:
            rank = Symbol(name='rank')
            rank_decl = LocalExpression(DummyEq(rank, 0))
            rank_init = Call('MPI_Comm_rank', [objcomm, Byref(rank)])

            ngpus = Symbol(name='ngpus')
            call = Function('omp_get_num_devices')()
            ngpus_init = LocalExpression(DummyEq(ngpus, call))

            osdd_then = Call('omp_set_default_device', [deviceid])
            osdd_else = Call('omp_set_default_device', [rank % ngpus])

            body = [Conditional(
                CondNe(deviceid, -1),
                osdd_then,
                List(body=[rank_decl, rank_init, ngpus_init, osdd_else]),
            )]
        else:
            body = [Conditional(
                CondNe(deviceid, -1),
                Call('omp_set_default_device', [deviceid])
            )]

        init = List(header=c.Comment('Begin of OpenMP+MPI setup'),
                    body=body,
                    footer=(c.Comment('End of OpenMP+MPI setup'), c.Line()))
        iet = iet._rebuild(body=(init,) + iet.body)

        return iet, {'args': deviceid}

    return _initialize(iet)


# Operators


class DeviceOpenMPOperatorMixin(object):

    _Parallelizer = DeviceOmpizer
    _DataManager = DeviceOpenMPDataManager
    _Initializer = initialize

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        oo = kwargs['options']

        oo.pop('openmp', None)  # It may or may not have been provided
        kwargs = super()._normalize_kwargs(**kwargs)
        oo['openmp'] = True

        return kwargs


class DeviceOpenMPNoopOperator(DeviceOpenMPOperatorMixin, DeviceNoopOperator):
    pass


class DeviceOpenMPOperator(DeviceOpenMPOperatorMixin, DeviceOperator):
    pass


class DeviceOpenMPCustomOperator(DeviceOpenMPOperatorMixin, DeviceCustomOperator):

    _known_passes = DeviceCustomOperator._known_passes + ('openmp',)
    assert not (set(_known_passes) & set(DeviceCustomOperator._known_passes_disabled))

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        mapper = super()._make_iet_passes_mapper(**kwargs)
        mapper['openmp'] = mapper['parallel']
        return mapper

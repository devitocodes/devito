from functools import partial, singledispatch

import cgen as c

from devito.core.gpu_openmp import (DeviceOpenMPNoopOperator, DeviceOpenMPCustomOperator,
                                    DeviceOpenMPIteration, DeviceOmpizer,
                                    DeviceOpenMPDataManager)
from devito.exceptions import InvalidOperator
from devito.ir.equations import DummyEq
from devito.ir.iet import Call, ElementalFunction, FindSymbols, List, LocalExpression
from devito.logger import warning
from devito.mpi.distributed import MPICommObject
from devito.mpi.routines import MPICallable
from devito.passes.iet import optimize_halospots, mpiize, hoist_prodders, iet_pass
from devito.symbolics import Byref, DefFunction, Macro
from devito.tools import as_tuple, prod, timed_pass
from devito.types import Symbol

__all__ = ['DeviceOpenACCNoopOperator', 'DeviceOpenACCOperator',
           'DeviceOpenACCCustomOperator']


# TODO: currently inhereting from the OpenMP Operators. Ideally, we should/could
# abstract things away so as to have a separate, language-agnostic superclass


class DeviceOpenACCIteration(DeviceOpenMPIteration):

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'acc parallel loop'

    @classmethod
    def _make_clauses(cls, **kwargs):
        kwargs['chunk_size'] = False
        clauses = super(DeviceOpenACCIteration, cls)._make_clauses(**kwargs)

        partree = kwargs['nodes']
        deviceptrs = [i.name for i in FindSymbols().visit(partree) if i.is_Array]
        if deviceptrs:
            clauses.append("deviceptr(%s)" % ",".join(deviceptrs))

        return clauses


class DeviceAccizer(DeviceOmpizer):

    lang = dict(DeviceOmpizer.__base__.lang)
    lang.update({
        'atomic': c.Pragma('acc atomic update'),
        'map-enter-to': lambda i, j:
            c.Pragma('acc enter data copyin(%s%s)' % (i, j)),
        'map-enter-alloc': lambda i, j:
            c.Pragma('acc enter data create(%s%s)' % (i, j)),
        'map-present': lambda i, j:
            c.Pragma('acc data present(%s%s)' % (i, j)),
        'map-update': lambda i, j:
            c.Pragma('acc exit data copyout(%s%s)' % (i, j)),
        'map-update-host': lambda i, j:
	    c.Pragma('acc update self(%s%s)' % (i, j)),
	'map-update-device': lambda i, j:
	    c.Pragma('acc update device(%s%s)' % (i, j)),
        'map-release': lambda i, j:
            c.Pragma('acc exit data delete(%s%s)' % (i, j)),
        'map-exit-delete': lambda i, j:
            c.Pragma('acc exit data delete(%s%s)' % (i, j)),
        'map-pointers': lambda i:
            c.Pragma('acc host_data use_device(%s)' % i)
    })

    _Iteration = DeviceOpenACCIteration

    @classmethod
    def _map_present(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-present'](f.name, sections)

    @classmethod
    def _map_delete(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-exit-delete'](f.name, sections)

    @classmethod
    def _map_pointers(cls, functions):
        return cls.lang['map-pointers'](','.join(f.name for f in functions))

    def _make_parallel(self, iet):
        iet, metadata = super(DeviceAccizer, self)._make_parallel(iet)

        metadata['includes'] = ['openacc.h']

        return iet, metadata


class DeviceOpenACCDataManager(DeviceOpenMPDataManager):

    _Parallelizer = DeviceAccizer

    def _alloc_array_on_high_bw_mem(self, site, obj, storage, memspace):
        """
        Allocate an Array in the high bandwidth memory.
        """
        size_trunkated = "".join("[%s]" % i for i in obj.symbolic_shape[1:])
        decl = c.Value(obj._C_typedata, "(*%s)%s" % (obj.name, size_trunkated))
        cast = "(%s (*)%s)" % (obj._C_typedata, size_trunkated)
        size_full = prod(obj.symbolic_shape)
        alloc = "%s acc_malloc(sizeof(%s[%s]))" % (cast, obj._C_typedata, size_full)
        init = c.Initializer(decl, alloc)

        free = c.Statement('acc_free(%s)' % obj.name)

        storage.update(obj, site, allocs=init, frees=free)


@iet_pass
def initialize(iet, **kwargs):
    """
    Initialize the OpenACC environment.
    """

    @singledispatch
    def _initialize(iet):
        # TODO: we need to pick the rank from `comm_shm`, not `comm`,
        # so that we have nranks == ngpus (as long as the user has launched
        # the right number of MPI processes per node given the available
        # number of GPUs per node)
        comm = None
        for i in iet.parameters:
            if isinstance(i, MPICommObject):
                comm = i
                break

        device_nvidia = Macro('acc_device_nvidia')
        body = Call('acc_init', [device_nvidia])

        if comm is not None:
            rank = Symbol(name='rank')
            rank_decl = LocalExpression(DummyEq(rank, 0))
            rank_init = Call('MPI_Comm_rank', [comm, Byref(rank)])

            ngpus = Symbol(name='ngpus')
            call = DefFunction('acc_get_num_devices', device_nvidia)
            ngpus_init = LocalExpression(DummyEq(ngpus, call))

            devicenum = Symbol(name='devicenum')
            devicenum_init = LocalExpression(DummyEq(devicenum, rank % ngpus))

            set_device_num = Call('acc_set_device_num', [devicenum, device_nvidia])

            body = [rank_decl, rank_init, ngpus_init, devicenum_init,
                    set_device_num, body]

        init = List(header=c.Comment('Begin of OpenACC+MPI setup'),
                    body=body,
                    footer=(c.Comment('End of OpenACC+MPI setup'), c.Line()))

        iet = iet._rebuild(body=(init,) + iet.body)

        return iet

    @_initialize.register(ElementalFunction)
    @_initialize.register(MPICallable)
    def _(iet):
        return iet

    iet = _initialize(iet)

    return iet, {}


class DeviceOpenACCNoopOperator(DeviceOpenMPNoopOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Device and host parallelism via OpenACC offloading
        accizer = DeviceAccizer(sregistry, options)
        accizer.make_parallel(graph)
        accizer.make_orchestration(graph)

        # Symbol definitions
        DeviceOpenACCDataManager(sregistry, options).process(graph)

        # Initialize OpenACC environment
        if options['mpi']:
            initialize(graph)

        return graph


class DeviceOpenACCOperator(DeviceOpenMPOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Device and host parallelism via OpenACC offloading
        accizer = DeviceAccizer(sregistry, options)
        accizer.make_parallel(graph)
        accizer.make_orchestration(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        DeviceOpenACCDataManager(sregistry, options).process(graph)

        # Initialize OpenACC environment
        if options['mpi']:
            initialize(graph)

        return graph


class DeviceOpenACCCustomOperator(DeviceOpenMPCustomOperator, DeviceOpenACCOperator):

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        accizer = DeviceAccizer(sregistry, options)

        return {
            'optcomms': partial(optimize_halospots),
            'openacc': partial(accizer.make_parallel),
            'orchestrate': partial(accizer.make_orchestration),
            'mpi': partial(mpiize, mode=options['mpi']),
            'prodders': partial(hoist_prodders)
        }

    _known_passes = (
        # Expressions
        'collect-deriv', 'buffering',
        # Clusters
        'blocking', 'tasking', 'fetching', 'factorize', 'fuse', 'lift',
        'cire-sops', 'cse', 'opt-pows', 'topofuse',
        # IET
        'optcomms', 'openacc', 'orchestrate', 'mpi', 'prodders'
    )
    _known_passes_disabled = ('openmp', 'denormals', 'simd', 'gpu-direct')
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

        # GPU parallelism via OpenACC offloading
        if 'openacc' not in passes:
            passes_mapper['openacc'](graph)

        # Call passes
        for i in passes:
            try:
                passes_mapper[i](graph)
            except KeyError:
                pass

        # Symbol definitions
        DeviceOpenACCDataManager(sregistry, options).process(graph)

        # Initialize OpenACC environment
        if options['mpi']:
            initialize(graph)

        return graph

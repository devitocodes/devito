from functools import partial, singledispatch

import cgen as c

from devito.core.gpu_openmp import (DeviceOpenMPNoopOperator, DeviceOpenMPOperator,
                                    DeviceOpenMPCustomOperator, DeviceOpenMPIteration,
                                    DeviceOmpizer, DeviceOpenMPDataManager, is_on_device)
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Conditional, EntryFunction, List, LocalExpression,
                           FindSymbols)
from devito.mpi.distributed import MPICommObject
from devito.passes.iet import (Orchestrator, optimize_halospots, mpiize, hoist_prodders,
                               iet_pass)
from devito.symbolics import Byref, CondNe, DefFunction, Macro
from devito.tools import as_tuple, prod, timed_pass
from devito.types import DeviceID, Symbol

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

        symbols = FindSymbols().visit(kwargs['nodes'])

        deviceptrs = [i.name for i in symbols if i.is_Array and i._mem_default]
        presents = [i.name for i in symbols
                    if (i.is_AbstractFunction and
                        is_on_device(i, kwargs['gpu_fit']) and
                        i.name not in deviceptrs)]

        # The NVC 20.7 and 20.9 compilers have a bug which triggers data movement for
        # indirectly indexed arrays (e.g., a[b[i]]) unless a present clause is used
        if presents:
            clauses.append("present(%s)" % ",".join(presents))

        if deviceptrs:
            clauses.append("deviceptr(%s)" % ",".join(deviceptrs))

        return clauses


class DeviceAccizer(DeviceOmpizer):

    lang = dict(DeviceOmpizer.__base__.lang)
    lang.update({
        'atomic': c.Pragma('acc atomic update'),
        'map-enter-to': lambda i, j:
            c.Pragma('acc enter data copyin(%s%s)' % (i, j)),
        'map-enter-to-wait': lambda i, j, k:
            (c.Pragma('acc enter data copyin(%s%s) async(%s)' % (i, j, k)),
             c.Pragma('acc wait(%s)' % k)),
        'map-enter-alloc': lambda i, j:
            c.Pragma('acc enter data create(%s%s)' % (i, j)),
        'map-present': lambda i, j:
            c.Pragma('acc data present(%s%s)' % (i, j)),
        'map-update': lambda i, j:
            c.Pragma('acc exit data copyout(%s%s)' % (i, j)),
        'map-update-host': lambda i, j:
            c.Pragma('acc update self(%s%s)' % (i, j)),
        'map-update-wait-host': lambda i, j, k:
            (c.Pragma('acc update self(%s%s) async(%s)' % (i, j, k)),
             c.Pragma('acc wait(%s)' % k)),
        'map-update-device': lambda i, j:
            c.Pragma('acc update device(%s%s)' % (i, j)),
        'map-update-wait-device': lambda i, j, k:
            (c.Pragma('acc update device(%s%s) async(%s)' % (i, j, k)),
             c.Pragma('acc wait(%s)' % k)),
        'map-release': lambda i, j, k:
            c.Pragma('acc exit data delete(%s%s)%s' % (i, j, k)),
        'map-exit-delete': lambda i, j, k:
            c.Pragma('acc exit data delete(%s%s)%s' % (i, j, k)),
        'map-pointers': lambda i:
            c.Pragma('acc host_data use_device(%s)' % i)
    })

    _Iteration = DeviceOpenACCIteration

    @classmethod
    def _map_to_wait(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-enter-to-wait'](f.name, sections, queueid)

    @classmethod
    def _map_present(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-present'](f.name, sections)

    @classmethod
    def _map_delete(cls, f, imask=None, devicerm=None):
        sections = cls._make_sections_from_imask(f, imask)
        if devicerm is not None:
            cond = ' if(%s)' % devicerm.name
        else:
            cond = ''
        return cls.lang['map-exit-delete'](f.name, sections, cond)

    @classmethod
    def _map_update_wait_host(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-update-wait-host'](f.name, sections, queueid)

    @classmethod
    def _map_update_wait_device(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-update-wait-device'](f.name, sections, queueid)

    @classmethod
    def _map_pointers(cls, functions):
        return cls.lang['map-pointers'](','.join(f.name for f in functions))

    def _make_parallel(self, iet):
        iet, metadata = super(DeviceAccizer, self)._make_parallel(iet)

        metadata['includes'] = ['openacc.h']

        return iet, metadata


class DeviceOpenACCOrchestrator(Orchestrator):

    _Parallelizer = DeviceAccizer


class DeviceOpenACCDataManager(DeviceOpenMPDataManager):

    _Parallelizer = DeviceAccizer

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
        """
        Allocate an Array in the high bandwidth memory.
        """
        if obj._mem_mapped:
            # posix_memalign + copy-to-device
            super()._alloc_array_on_high_bw_mem(site, obj, storage)
        else:
            # acc_malloc -- the Array only resides on the device, ie, it never
            # needs to be accessed on the host
            assert obj._mem_default
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
        device_nvidia = Macro('acc_device_nvidia')
        if objcomm is not None:
            rank = Symbol(name='rank')
            rank_decl = LocalExpression(DummyEq(rank, 0))
            rank_init = Call('MPI_Comm_rank', [objcomm, Byref(rank)])

            ngpus = Symbol(name='ngpus')
            call = DefFunction('acc_get_num_devices', device_nvidia)
            ngpus_init = LocalExpression(DummyEq(ngpus, call))

            asdn_then = Call('acc_set_device_num', [deviceid, device_nvidia])
            asdn_else = Call('acc_set_device_num', [rank % ngpus, device_nvidia])

            body = [Call('acc_init', [device_nvidia]), Conditional(
                CondNe(deviceid, -1),
                asdn_then,
                List(body=[rank_decl, rank_init, ngpus_init, asdn_else])
            )]
        else:
            body = [Call('acc_init', [device_nvidia]), Conditional(
                CondNe(deviceid, -1),
                Call('acc_set_device_num', [deviceid, device_nvidia])
            )]

        init = List(header=c.Comment('Begin of OpenACC+MPI setup'),
                    body=body,
                    footer=(c.Comment('End of OpenACC+MPI setup'), c.Line()))
        iet = iet._rebuild(body=(init,) + iet.body)

        return iet, {'args': deviceid}

    return _initialize(iet)


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

        # Symbol definitions
        DeviceOpenACCDataManager(sregistry, options).process(graph)

        # Initialize OpenACC environment
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

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        DeviceOpenACCDataManager(sregistry, options).process(graph)

        # Initialize OpenACC environment
        initialize(graph)

        return graph


class DeviceOpenACCCustomOperator(DeviceOpenMPCustomOperator, DeviceOpenACCOperator):

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        accizer = DeviceAccizer(sregistry, options)
        orchestrator = DeviceOpenACCOrchestrator(sregistry)

        return {
            'optcomms': partial(optimize_halospots),
            'openacc': partial(accizer.make_parallel),
            'orchestrate': partial(orchestrator.process),
            'mpi': partial(mpiize, mode=options['mpi']),
            'prodders': partial(hoist_prodders)
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
        initialize(graph)

        return graph

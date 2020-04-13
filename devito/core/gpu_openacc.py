from functools import partial, singledispatch

import cgen as c
from sympy import Function

from devito.core.gpu_openmp import (DeviceOpenMPNoopOperator, DeviceOpenMPIteration,
                                    DeviceOmpizer, DeviceOpenMPDataManager)
from devito.exceptions import InvalidOperator
from devito.ir.equations import DummyEq
from devito.ir.iet import Call, ElementalFunction, FindSymbols, List, LocalExpression
from devito.logger import warning
from devito.mpi.distributed import MPICommObject
from devito.mpi.routines import MPICallable
from devito.passes.iet import optimize_halospots, mpiize, hoist_prodders, iet_pass
from devito.symbolics import Byref, Macro
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

    lang = {
        'atomic': c.Pragma('acc atomic update'),
        'map-enter-to': lambda i, j:
            c.Pragma('acc enter data copyin(%s%s)' % (i, j)),
        'map-enter-alloc': lambda i, j:
            c.Pragma('acc enter data create(%s%s)' % (i, j)),
        'map-present': lambda i, j:
            c.Pragma('acc data present(%s%s)' % (i, j)),
        'map-update': lambda i, j:
            c.Pragma('acc exit data copyout(%s%s)' % (i, j)),
        'map-release': lambda i, j:
            c.Pragma('acc exit data delete(%s%s)' % (i, j)),
        'map-exit-delete': lambda i, j:
            c.Pragma('acc exit data delete(%s%s)' % (i, j)),
        'map-pointers': lambda i:
            c.Pragma('acc host_data use_device(%s)' % i)
    }

    _Iteration = DeviceOpenACCIteration

    @classmethod
    def _map_present(cls, f):
        # TODO: currently this is unused, because we cannot yet distinguish between
        # "real" Arrays and Functions that "acts as Arrays", created by the compiler
        # to build support routines (e.g., the Sendrecv/Gather/Scatter MPI Callables).
        # We should only use "#pragma acc present" for *real* Arrays -- that is
        # temporaries that are born and die on the Device
        return cls.lang['map-present'](f.name, ''.join('[0:%s]' % i
                                                       for i in cls._map_data(f)))

    @classmethod
    def _map_pointers(cls, functions):
        return cls.lang['map-pointers'](','.join(f.name for f in functions))

    def _make_parallel(self, iet):
        iet, metadata = super(DeviceAccizer, self)._make_parallel(iet)

        metadata['includes'] = ['openacc.h']

        return iet, metadata


class DeviceOpenACCDataManager(DeviceOpenMPDataManager):

    _Parallelizer = DeviceAccizer

    def _alloc_array_on_high_bw_mem(self, obj, storage):
        """Allocate an Array in the high bandwidth memory."""
        if obj in storage._high_bw_mem:
            return

        size_trunkated = "".join("[%s]" % i for i in obj.symbolic_shape[1:])
        decl = c.Value(obj._C_typedata, "(*%s)%s" % (obj.name, size_trunkated))
        cast = "(%s (*)%s)" % (obj._C_typedata, size_trunkated)
        size_full = prod(obj.symbolic_shape)
        alloc = "%s acc_malloc(sizeof(%s[%s]))" % (cast, obj._C_typedata, size_full)
        init = c.Initializer(decl, alloc)

        free = c.Statement('acc_free(%s)' % obj.name)

        storage._high_bw_mem[obj] = (None, init, free)


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
            call = Function('acc_get_num_devices')(device_nvidia)
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

        # Distributed-memory parallelism
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # GPU parallelism via OpenACC offloading
        DeviceAccizer().make_parallel(graph)

        # Symbol definitions
        data_manager = DeviceOpenACCDataManager()
        data_manager.place_ondevice(graph)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        # Initialize OpenACC environment
        if options['mpi']:
            initialize(graph)

        return graph


class DeviceOpenACCOperator(DeviceOpenACCNoopOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # GPU parallelism via OpenACC offloading
        DeviceAccizer().make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DeviceOpenACCDataManager()
        data_manager.place_ondevice(graph)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        # Initialize OpenACC environment
        if options['mpi']:
            initialize(graph)

        return graph


class DeviceOpenACCCustomOperator(DeviceOpenACCOperator):

    _known_passes = ('optcomms', 'openacc', 'mpi', 'prodders')
    _known_passes_disabled = ('blocking', 'openmp', 'denormals', 'wrapping', 'simd')
    assert not (set(_known_passes) & set(_known_passes_disabled))

    @classmethod
    def _make_passes_mapper(cls, **kwargs):
        options = kwargs['options']

        accizer = DeviceAccizer()

        return {
            'optcomms': partial(optimize_halospots),
            'openacc': partial(accizer.make_parallel),
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

        return super(DeviceOpenACCCustomOperator, cls)._build(expressions, **kwargs)

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

        # GPU parallelism via OpenACC offloading
        if 'openacc' not in passes:
            passes_mapper['openacc'](graph)

        # Symbol definitions
        data_manager = DeviceOpenACCDataManager()
        data_manager.place_ondevice(graph)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        # Initialize OpenACC environment
        if options['mpi']:
            initialize(graph)

        return graph

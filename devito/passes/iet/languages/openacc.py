from functools import singledispatch

import cgen as c

from devito.ir import (DummyEq, Call, Conditional, EntryFunction, List, LocalExpression,
                       ParallelIteration, FindSymbols)
from devito.mpi.distributed import MPICommObject
from devito.passes.iet.definitions import DeviceAwareDataManager
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.parpragma import PragmaDeviceAwareTransformer, PragmaLangBB
from devito.passes.iet.languages.openmp import OmpRegion, OmpIteration
from devito.passes.iet.languages.utils import make_clause_reduction
from devito.passes.iet.misc import is_on_device
from devito.symbolics import Byref, CondNe, DefFunction, Macro
from devito.tools import prod
from devito.types import DeviceID, Symbol

__all__ = ['DeviceAccizer', 'DeviceAccDataManager', 'AccOrchestrator']


class DeviceAccIteration(ParallelIteration):

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'acc parallel loop'

    @classmethod
    def _make_clauses(cls, ncollapse=None, reduction=None, **kwargs):
        clauses = []

        clauses.append('collapse(%d)' % (ncollapse or 1))

        if reduction:
            clauses.append(make_clause_reduction(reduction))

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

    @classmethod
    def _process_kwargs(cls, **kwargs):
        kwargs = super()._process_kwargs(**kwargs)

        kwargs.pop('gpu_fit', None)

        kwargs.pop('schedule', None)
        kwargs.pop('parallel', False)
        kwargs.pop('chunk_size', None)
        kwargs.pop('nthreads', None)

        return kwargs


class AccBB(PragmaLangBB):

    mapper = {
        'header': 'openacc.h',
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
        'device-alloc': lambda i:
            'acc_malloc(%s)' % i,
        'device-free': lambda i:
            'acc_free(%s)' % i
    }

    Region = OmpRegion
    HostIteration = OmpIteration  # Host parallelism still goes via OpenMP
    DeviceIteration = DeviceAccIteration

    @classmethod
    def _map_to_wait(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-enter-to-wait'](f.name, sections, queueid)

    @classmethod
    def _map_present(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-present'](f.name, sections)

    @classmethod
    def _map_delete(cls, f, imask=None, devicerm=None):
        sections = cls._make_sections_from_imask(f, imask)
        if devicerm is not None:
            cond = ' if(%s)' % devicerm.name
        else:
            cond = ''
        return cls.mapper['map-exit-delete'](f.name, sections, cond)

    @classmethod
    def _map_update_wait_host(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-update-wait-host'](f.name, sections, queueid)

    @classmethod
    def _map_update_wait_device(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-update-wait-device'](f.name, sections, queueid)


class DeviceAccizer(PragmaDeviceAwareTransformer):

    lang = AccBB

    @iet_pass
    def make_gpudirect(self, iet):
        # Implicitly handled since acc_malloc is used, hence device pointers
        # are passed to MPI calls
        return iet, {}

    @iet_pass
    def initialize(self, iet):
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


class DeviceAccDataManager(DeviceAwareDataManager):

    lang = AccBB

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
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
            size_full = "sizeof(%s[%s])" % (obj._C_typedata, prod(obj.symbolic_shape))
            alloc = "%s %s" % (cast, self.lang['device-alloc'](size_full))
            init = c.Initializer(decl, alloc)

            free = c.Statement(self.lang['device-free'](obj.name))

            storage.update(obj, site, allocs=init, frees=free)


class AccOrchestrator(Orchestrator):
    lang = AccBB

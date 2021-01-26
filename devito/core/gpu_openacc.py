from functools import singledispatch

import cgen as c

from devito.core.gpu import DeviceNoopOperator, DeviceOperator, DeviceCustomOperator
from devito.core.gpu_openmp import DeviceOmpDataManager
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Conditional, EntryFunction, List, LocalExpression,
                           FindSymbols)
from devito.mpi.distributed import MPICommObject
from devito.passes.iet import DeviceOmpizer, DeviceOmpIteration, iet_pass, is_on_device
from devito.symbolics import Byref, CondNe, DefFunction, Macro
from devito.tools import prod
from devito.types import DeviceID, Symbol

__all__ = ['DeviceNoopAccOperator', 'DeviceAdvAccOperator', 'DeviceCustomAccOperator']


class DeviceAccIteration(DeviceOmpIteration):

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'acc parallel loop'

    @classmethod
    def _make_clauses(cls, **kwargs):
        kwargs['chunk_size'] = False
        clauses = super(DeviceAccIteration, cls)._make_clauses(**kwargs)

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
        'header': 'openacc.h'
    })

    _Iteration = DeviceAccIteration

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


class DeviceAccDataManager(DeviceOmpDataManager):

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


# Operators


class DeviceAccOperatorMixin(object):

    _Parallelizer = DeviceAccizer
    _DataManager = DeviceAccDataManager

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        oo = kwargs['options']
        oo.pop('openmp', None)

        kwargs = super()._normalize_kwargs(**kwargs)
        oo['openacc'] = True

        return kwargs


class DeviceNoopAccOperator(DeviceAccOperatorMixin, DeviceNoopOperator):
    pass


class DeviceAdvAccOperator(DeviceAccOperatorMixin, DeviceOperator):
    pass


class DeviceCustomAccOperator(DeviceAccOperatorMixin, DeviceCustomOperator):

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        mapper = super()._make_iet_passes_mapper(**kwargs)
        mapper['openacc'] = mapper['parallel']
        return mapper

    _known_passes = DeviceCustomOperator._known_passes + ('openacc',)
    assert not (set(_known_passes) & set(DeviceCustomOperator._known_passes_disabled))

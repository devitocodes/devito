import cgen as c

from devito.arch import AMDGPUX, NVIDIAX
from devito.ir import Call, List, ParallelIteration, ParallelTree, FindSymbols
from devito.passes.iet.definitions import DeviceAwareDataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.parpragma import PragmaDeviceAwareTransformer, PragmaLangBB
from devito.passes.iet.languages.C import CBB
from devito.passes.iet.languages.openmp import OmpRegion, OmpIteration
from devito.passes.iet.languages.utils import make_clause_reduction
from devito.passes.iet.misc import is_on_device
from devito.symbolics import DefFunction, Macro
from devito.tools import filter_ordered, prod

__all__ = ['DeviceAccizer', 'DeviceAccDataManager', 'AccOrchestrator']


class DeviceAccIteration(ParallelIteration):

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'acc parallel loop'

    @classmethod
    def _make_clauses(cls, ncollapse=None, reduction=None, tile=None, **kwargs):
        clauses = []

        if ncollapse:
            clauses.append('collapse(%d)' % (ncollapse or 1))
        elif tile:
            clauses.append('tile(%s)' % ','.join(str(i) for i in tile))

        if reduction:
            clauses.append(make_clause_reduction(reduction))

        indexeds = FindSymbols('indexeds').visit(kwargs['nodes'])
        deviceptrs = filter_ordered(i.name for i in indexeds
                                    if i.function.is_Array and i.function._mem_default)
        presents = filter_ordered(i.name for i in indexeds
                                  if (i.function.is_AbstractFunction and
                                      is_on_device(i, kwargs['gpu_fit']) and
                                      i.name not in deviceptrs))

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
        kwargs.pop('parallel', None)
        kwargs.pop('chunk_size', None)
        kwargs.pop('nthreads', None)
        kwargs.pop('tile', None)

        return kwargs


class AccBB(PragmaLangBB):

    mapper = {
        # Misc
        'name': 'OpenACC',
        'header': 'openacc.h',
        # Platform mapping
        AMDGPUX: Macro('acc_device_radeon'),
        NVIDIAX: Macro('acc_device_nvidia'),
        # Runtime library
        'init': lambda args:
            Call('acc_init', args),
        'num-devices': lambda args:
            DefFunction('acc_get_num_devices', args),
        'set-device': lambda args:
            Call('acc_set_device_num', args),
        # Pragmas
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
        'memcpy-to-device': lambda i, j, k:
            Call('acc_memcpy_to_device', [i, j, k]),
        'memcpy-to-device-wait': lambda i, j, k, l:
            List(body=[Call('acc_memcpy_to_device_async', [i, j, k, l]),
                       Call('acc_wait', [l])]),
        'device-alloc': lambda i:
            'acc_malloc(%s)' % i,
        'device-free': lambda i:
            'acc_free(%s)' % i
    }
    mapper.update(CBB.mapper)

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

    # Note: there is no need to override `make_gpudirect` since acc_malloc is
    # used to allocate the buffers passed to the various MPI calls, which will
    # then receive device points

    def _make_partree(self, candidates, nthreads=None):
        assert candidates
        root = candidates[0]

        collapsable = self._find_collapsable(root, candidates)
        ncollapsable = len(collapsable)

        if self._is_offloadable(root) and \
           all(i.is_Affine for i in [root] + collapsable) and \
           self.par_tile:
            if isinstance(self.par_tile, tuple):
                tile = self.par_tile[:ncollapsable + 1]
            else:
                # (32,4,4,...) is typically a decent choice
                tile = (32,) + (4,)*ncollapsable

            body = self.DeviceIteration(gpu_fit=self.gpu_fit, tile=tile, **root.args)
            partree = ParallelTree([], body, nthreads=nthreads)

            return root, partree
        else:
            return super()._make_partree(candidates, nthreads)


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

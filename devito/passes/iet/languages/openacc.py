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
from devito.tools import filter_ordered

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
        deviceptrs = filter_ordered(i.name for i in indexeds if i.function._mem_local)
        presents = filter_ordered(i.name for i in indexeds
                                  if (is_on_device(i, kwargs['gpu_fit']) and
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
        'map-wait': lambda i:
            c.Pragma('acc wait(%s)' % i),
        'map-update': lambda i, j:
            c.Pragma('acc exit data copyout(%s%s)' % (i, j)),
        'map-update-host': lambda i, j:
            c.Pragma('acc update self(%s%s)' % (i, j)),
        'map-update-host-async': lambda i, j, k:
            c.Pragma('acc update self(%s%s) async(%s)' % (i, j, k)),
        'map-update-device': lambda i, j:
            c.Pragma('acc update device(%s%s)' % (i, j)),
        'map-update-device-async': lambda i, j, k:
            c.Pragma('acc update device(%s%s) async(%s)' % (i, j, k)),
        'map-release': lambda i, j, k:
            c.Pragma('acc exit data delete(%s%s)%s' % (i, j, k)),
        'map-exit-delete': lambda i, j, k:
            c.Pragma('acc exit data delete(%s%s)%s' % (i, j, k)),
        'memcpy-to-device': lambda i, j, k:
            Call('acc_memcpy_to_device', [i, j, k]),
        'memcpy-to-device-wait': lambda i, j, k, l:
            List(body=[Call('acc_memcpy_to_device_async', [i, j, k, l]),
                       Call('acc_wait', [l])]),
        'device-get':
            Call('acc_get_device_num'),
        'device-alloc': lambda i, *a, retobj=None:
            Call('acc_malloc', (i,), retobj=retobj, cast=True),
        'device-free': lambda i, *a:
            Call('acc_free', (i,))
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
    def _map_update_host_async(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-update-host-async'](f.name, sections, queueid)

    @classmethod
    def _map_update_device_async(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-update-device-async'](f.name, sections, queueid)


class DeviceAccizer(PragmaDeviceAwareTransformer):

    lang = AccBB

    def _make_partree(self, candidates, nthreads=None):
        assert candidates

        root, collapsable = self._select_candidates(candidates)
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


class AccOrchestrator(Orchestrator):
    lang = AccBB

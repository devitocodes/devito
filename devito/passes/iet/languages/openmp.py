import cgen as c
from sympy import Not

from devito.arch import AMDGPUX, NVIDIAX
from devito.ir import (Call, Conditional, List, Prodder, ParallelIteration,
                       ParallelBlock, PointerCast, While, FindSymbols)
from devito.passes.iet.definitions import DataManager, DeviceAwareDataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.parpragma import (PragmaSimdTransformer, PragmaShmTransformer,
                                         PragmaDeviceAwareTransformer, PragmaLangBB)
from devito.passes.iet.languages.C import CBB
from devito.passes.iet.languages.utils import make_clause_reduction
from devito.symbolics import CondEq, DefFunction
from devito.tools import filter_ordered

__all__ = ['SimdOmpizer', 'Ompizer', 'OmpIteration', 'OmpRegion',
           'DeviceOmpizer', 'DeviceOmpIteration', 'DeviceOmpDataManager',
           'OmpDataManager', 'OmpOrchestrator']


class OmpRegion(ParallelBlock):

    @classmethod
    def _make_header(cls, nthreads, private=None):
        private = ('private(%s)' % ','.join(private)) if private else ''
        return c.Pragma('omp parallel num_threads(%s) %s' % (nthreads.name, private))


class OmpIteration(ParallelIteration):

    @classmethod
    def _make_construct(cls, parallel=False, **kwargs):
        if parallel:
            return 'omp parallel for'
        else:
            return 'omp for'

    @classmethod
    def _make_clauses(cls, ncollapse=None, chunk_size=None, nthreads=None,
                      reduction=None, schedule=None, **kwargs):
        clauses = []

        clauses.append('collapse(%d)' % (ncollapse or 1))

        if chunk_size is not False:
            clauses.append('schedule(%s,%s)' % (schedule or 'dynamic',
                                                chunk_size or 1))

        if nthreads:
            clauses.append('num_threads(%s)' % nthreads)

        if reduction:
            clauses.append(make_clause_reduction(reduction))

        return clauses

    @classmethod
    def _process_kwargs(cls, **kwargs):
        kwargs = super()._process_kwargs(**kwargs)

        kwargs.pop('schedule', None)
        kwargs.pop('parallel', False)
        kwargs.pop('chunk_size', None)
        kwargs.pop('nthreads', None)

        return kwargs


class DeviceOmpIteration(OmpIteration):

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'omp target teams distribute parallel for'

    @classmethod
    def _make_clauses(cls, **kwargs):
        kwargs['chunk_size'] = False
        clauses = super()._make_clauses(**kwargs)

        indexeds = FindSymbols('indexeds').visit(kwargs['nodes'])
        deviceptrs = filter_ordered(i.name for i in indexeds if i.function._mem_local)
        if deviceptrs:
            clauses.append("is_device_ptr(%s)" % ",".join(deviceptrs))

        return clauses

    @classmethod
    def _process_kwargs(cls, **kwargs):
        kwargs = super()._process_kwargs(**kwargs)

        kwargs.pop('gpu_fit', None)

        return kwargs


class ThreadedProdder(Conditional, Prodder):

    _traversable = []

    def __init__(self, prodder):
        # Atomic-ize any single-thread Prodders in the parallel tree
        condition = CondEq(DefFunction(Ompizer.lang['thread-num']().name), 0)

        # Prod within a while loop until all communications have completed
        # In other words, the thread delegated to prodding is entrapped for as long
        # as it's required
        prod_until = Not(DefFunction(prodder.name, [i.name for i in prodder.arguments]))
        then_body = List(header=c.Comment('Entrap thread until comms have completed'),
                         body=While(prod_until))

        Conditional.__init__(self, condition, then_body)
        Prodder.__init__(self, prodder.name, prodder.arguments, periodic=prodder.periodic)


class OmpBB(PragmaLangBB):

    mapper = {
        # Misc
        'name': 'OpenMP',
        'header': 'omp.h',
        # Platform mapping
        AMDGPUX: None,
        NVIDIAX: None,
        # Runtime library
        'init': None,
        'thread-num': lambda retobj=None:
            Call('omp_get_thread_num', retobj=retobj),
        'num-devices': lambda args:
            DefFunction('omp_get_num_devices', args),
        'set-device': lambda args:
            Call('omp_set_default_device', args),
        # Pragmas
        'simd-for': c.Pragma('omp simd'),
        'simd-for-aligned': lambda i, j: c.Pragma('omp simd aligned(%s:%d)' % (i, j)),
        'atomic': c.Pragma('omp atomic update'),
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
        'memcpy-to-device': lambda i, j, k:
            Call('omp_target_memcpy', [i, j, k, 0, 0,
                                       DefFunction('omp_get_device_num'),
                                       DefFunction('omp_get_initial_device')]),
        'memcpy-to-device-wait': lambda i, j, k, l:
            Call('omp_target_memcpy', [i, j, k, 0, 0,
                                       DefFunction('omp_get_device_num'),
                                       DefFunction('omp_get_initial_device')]),
        'device-get':
            Call('omp_get_default_device'),
        'device-alloc': lambda i, j, *a, **kw:
            Call('omp_target_alloc', (i, j)),
        'device-free': lambda i, j:
            Call('omp_target_free', (i, j))
    }
    mapper.update(CBB.mapper)

    Region = OmpRegion
    HostIteration = OmpIteration
    DeviceIteration = DeviceOmpIteration
    Prodder = ThreadedProdder


class DeviceOmpBB(OmpBB):

    # NOTE: Work around clang>=10 issue concerning offloading arrays declared
    # with an `__attribute__(aligned(...))` qualifier
    PointerCast = lambda *a, **kw: PointerCast(*a, alignment=False, **kw)


class SimdOmpizer(PragmaSimdTransformer):
    lang = OmpBB


class Ompizer(PragmaShmTransformer):
    lang = OmpBB


class DeviceOmpizer(PragmaDeviceAwareTransformer):
    lang = DeviceOmpBB


class OmpDataManager(DataManager):
    lang = OmpBB


class DeviceOmpDataManager(DeviceAwareDataManager):
    lang = DeviceOmpBB


class OmpOrchestrator(Orchestrator):
    lang = DeviceOmpBB

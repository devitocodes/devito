from packaging.version import Version

import cgen as c
from sympy import And, Ne, Not

from devito.arch import AMDGPUX, NVIDIAX, INTELGPUX
from devito.arch.compiler import GNUCompiler
from devito.ir import (Call, Conditional, DeviceCall, List, Prodder,
                       ParallelIteration, ParallelBlock, PointerCast, While,
                       FindSymbols)
from devito.passes.iet.definitions import DataManager, DeviceAwareDataManager
from devito.passes.iet.langbase import LangBB
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.parpragma import (PragmaSimdTransformer, PragmaShmTransformer,
                                         PragmaDeviceAwareTransformer, PragmaLangBB,
                                         PragmaTransfer)
from devito.passes.iet.languages.C import CBB
from devito.passes.iet.languages.utils import make_clause_reduction
from devito.symbolics import CondEq, DefFunction
from devito.tools import filter_ordered

__all__ = ['SimdOmpizer', 'Ompizer', 'OmpIteration', 'OmpRegion',
           'DeviceOmpizer', 'DeviceOmpIteration', 'DeviceOmpDataManager',
           'OmpDataManager', 'OmpOrchestrator', 'DeviceOmpOrchestrator']


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

    def __init__(self, prodder, arguments=None):
        # Atomic-ize any single-thread Prodders in the parallel tree
        condition = CondEq(DefFunction(Ompizer.lang['thread-num']().name), 0)

        # Prod within a while loop until all communications have completed
        # In other words, the thread delegated to prodding is entrapped for as long
        # as it's required
        prod_until = Not(DefFunction(prodder.name, [i.name for i in prodder.arguments]))
        then_body = List(header=c.Comment('Entrap thread until comms have completed'),
                         body=While(prod_until))
        Conditional.__init__(self, condition, then_body)

        arguments = arguments or prodder.arguments
        Prodder.__init__(self, prodder.name, arguments, periodic=prodder.periodic)


class OmpBB(LangBB):

    mapper = {
        # Misc
        'name': 'OpenMP',
        'header': 'omp.h',
        # Platform mapping
        AMDGPUX: None,
        NVIDIAX: None,
        INTELGPUX: None,
        # Runtime library
        'init': None,
        'thread-num': lambda retobj=None:
            Call('omp_get_thread_num', retobj=retobj),
        # Pragmas
        'simd-for': c.Pragma('omp simd'),
        'simd-for-aligned': lambda i, j: c.Pragma('omp simd aligned(%s:%d)' % (i, j)),
        'atomic': c.Pragma('omp atomic update')
    }
    mapper.update(CBB.mapper)

    Region = OmpRegion
    HostIteration = OmpIteration
    DeviceIteration = DeviceOmpIteration
    Prodder = ThreadedProdder


class DeviceOmpBB(OmpBB, PragmaLangBB):

    BackendCall = DeviceCall

    mapper = dict(OmpBB.mapper)
    mapper.update({
        # Runtime library
        'num-devices': lambda args, retobj:
            Call('omp_get_num_devices', args, retobj=retobj),
        'set-device': lambda args:
            Call('omp_set_default_device', args),
        # Pragmas
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
            c.Pragma('omp target exit data map(release: %s%s)' % (i, j)),
        'map-release-if': lambda i, j, k:
            c.Pragma('omp target exit data map(release: %s%s) if(%s)' % (i, j, k)),
        'map-exit-delete': lambda i, j:
            c.Pragma('omp target exit data map(delete: %s%s)' % (i, j)),
        'map-exit-delete-if': lambda i, j, k:
            c.Pragma('omp target exit data map(delete: %s%s) if(%s)' % (i, j, k)),
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
        'device-alloc': lambda i, j, retobj:
            Call('omp_target_alloc', (i, j), retobj=retobj, cast=True),
        'device-free': lambda i, j:
            Call('omp_target_free', (i, j))
    })

    # NOTE: Work around clang>=10 issue concerning offloading arrays declared
    # with an `__attribute__(aligned(...))` qualifier
    PointerCast = lambda *a, **kw: PointerCast(*a, alignment=False, **kw)

    @classmethod
    def _map_delete(cls, f, imask=None, devicerm=None):
        # This ugly condition is to avoid a copy-back when, due to
        # domain decomposition, the local size of a Function is 0, which
        # would cause a crash with some OpenMP-offloading implementations
        items = [Ne(i, 0, evaluate=False) for i in f.symbolic_shape]
        if devicerm is not None:
            items.append(devicerm)
        argument = And(*items)
        return PragmaTransfer(cls.mapper['map-exit-delete-if'], f, imask, argument)


class SimdOmpizer(PragmaSimdTransformer):
    lang = OmpBB


class Ompizer(PragmaShmTransformer):

    lang = OmpBB

    @classmethod
    def _support_array_reduction(cls, compiler):
        # Not all backend compilers support array reduction!
        # Here are the known unsupported ones:
        if isinstance(compiler, GNUCompiler) and \
           compiler.version < Version("6.0"):
            return False
        return True


class DeviceOmpizer(PragmaDeviceAwareTransformer):
    lang = DeviceOmpBB


class OmpDataManager(DataManager):
    lang = OmpBB


class DeviceOmpDataManager(DeviceAwareDataManager):
    lang = DeviceOmpBB


class OmpOrchestrator(Orchestrator):
    lang = OmpBB


class DeviceOmpOrchestrator(Orchestrator):
    lang = DeviceOmpBB

from functools import singledispatch

import cgen as c
from sympy import Function, Not

from devito.ir import (DummyEq, Call, Conditional, List, Prodder, ParallelIteration,
                       ParallelBlock, ParallelTree, EntryFunction, LocalExpression,
                       While)
from devito.mpi.distributed import MPICommObject
from devito.passes.iet.definitions import DataManager, DeviceAwareDataManager
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.parpragma import (PragmaSimdTransformer, PragmaShmTransformer,
                                         PragmaDeviceAwareTransformer, PragmaLangBB)
from devito.passes.iet.languages.C import CBB
from devito.passes.iet.languages.utils import make_clause_reduction
from devito.symbolics import Byref, CondEq, CondNe, DefFunction
from devito.tools import as_tuple
from devito.types import DeviceID, Symbol

__all__ = ['SimdOmpizer', 'Ompizer', 'OmpIteration', 'OmpRegion',
           'DeviceOmpizer', 'DeviceOmpIteration', 'DeviceOmpDataManager',
           'OmpDataManager', 'OmpOrchestrator']


class OmpRegion(ParallelBlock):

    def __init__(self, body, private=None):
        # Normalize and sanity-check input. A bit ugly, but it makes everything
        # much simpler to manage and reconstruct
        body = as_tuple(body)
        assert len(body) == 1
        body = body[0]
        assert body.is_List
        if isinstance(body, ParallelTree):
            partree = body
        elif body.is_List:
            assert len(body.body) == 1 and isinstance(body.body[0], ParallelTree)
            assert len(body.footer) == 0
            partree = body.body[0]
            partree = partree._rebuild(prefix=(List(header=body.header,
                                                    body=partree.prefix)))

        header = OmpRegion._make_header(partree.nthreads, private)
        super().__init__(header=header, body=partree)

    @property
    def partree(self):
        return self.body[0]

    @property
    def root(self):
        return self.partree.root

    @property
    def nthreads(self):
        return self.partree.nthreads

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
        return super()._make_clauses(**kwargs)

    @classmethod
    def _process_kwargs(cls, **kwargs):
        kwargs = super()._process_kwargs(**kwargs)

        kwargs.pop('gpu_fit', None)

        return kwargs


class ThreadedProdder(Conditional, Prodder):

    _traversable = []

    def __init__(self, prodder):
        # Atomic-ize any single-thread Prodders in the parallel tree
        condition = CondEq(Ompizer.lang['thread-num'], 0)

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
        'header': 'omp.h',
        'simd-for': c.Pragma('omp simd'),
        'simd-for-aligned': lambda i, j: c.Pragma('omp simd aligned(%s:%d)' % (i, j)),
        'atomic': c.Pragma('omp atomic update'),
        'thread-num': DefFunction('omp_get_thread_num'),
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
    }
    mapper.update(CBB.mapper)

    Region = OmpRegion
    HostIteration = OmpIteration
    DeviceIteration = DeviceOmpIteration
    Prodder = ThreadedProdder


class SimdOmpizer(PragmaSimdTransformer):
    lang = OmpBB


class Ompizer(PragmaShmTransformer):
    lang = OmpBB


class DeviceOmpizer(PragmaDeviceAwareTransformer):

    lang = OmpBB

    @iet_pass
    def initialize(self, iet):

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


class OmpDataManager(DataManager):
    lang = OmpBB


class DeviceOmpDataManager(DeviceAwareDataManager):
    lang = OmpBB


class OmpOrchestrator(Orchestrator):
    lang = OmpBB

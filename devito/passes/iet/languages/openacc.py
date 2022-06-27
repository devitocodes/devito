import cgen as c
import numpy as np

from devito.arch import AMDGPUX, NVIDIAX
from devito.ir import (Call, DeviceCall, DummyExpr, DPtr, EntryFunction, List,
                       Block, ParallelIteration, ParallelTree, Pragma,
                       FindNodes, FindSymbols, Uxreplace, Transformer)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.parpragma import (PragmaDeviceAwareTransformer, PragmaLangBB,
                                         PragmaTransfer, PragmaDeviceAwareDataManager)
from devito.passes.iet.languages.C import CBB
from devito.passes.iet.languages.openmp import OmpRegion, OmpIteration
from devito.passes.iet.languages.utils import make_clause_reduction
from devito.passes.iet.misc import is_on_device
from devito.symbolics import Macro, cast_mapper
from devito.tools import filter_ordered
from devito.types import DevicePointer, Symbol

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
        'num-devices': lambda args, retobj:
            Call('acc_get_num_devices', args, retobj=retobj),
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
        'map-release': lambda i, j:
            c.Pragma('acc exit data delete(%s%s)' % (i, j)),
        'map-release-if': lambda i, j, k:
            c.Pragma('acc exit data delete(%s%s) if(%s)' % (i, j, k)),
        'map-exit-delete': lambda i, j:
            c.Pragma('acc exit data delete(%s%s)' % (i, j)),
        'map-exit-delete-if': lambda i, j, k:
            c.Pragma('acc exit data delete(%s%s) if(%s)' % (i, j, k)),
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
    def _map_to_wait(cls, f, imask=None, qid=None):
        return PragmaTransfer(cls.mapper['map-enter-to-wait'], f, imask, qid)

    @classmethod
    def _map_present(cls, f, imask=None):
        return PragmaTransfer(cls.mapper['map-present'], f, imask)

    @classmethod
    def _map_wait(cls, qid=None):
        return Pragma(cls.mapper['map-wait'], qid)

    @classmethod
    def _map_delete(cls, f, imask=None, devicerm=None):
        if devicerm:
            return PragmaTransfer(cls.mapper['map-exit-delete-if'], f, imask, devicerm)
        else:
            return PragmaTransfer(cls.mapper['map-exit-delete'], f, imask)

    @classmethod
    def _map_update_host_async(cls, f, imask=None, qid=None):
        return PragmaTransfer(cls.mapper['map-update-host-async'], f, imask, qid)

    @classmethod
    def _map_update_device_async(cls, f, imask=None, qid=None):
        return PragmaTransfer(cls.mapper['map-update-device-async'], f, imask, qid)


class DeviceAccizer(PragmaDeviceAwareTransformer):

    lang = AccBB

    def _make_partree(self, candidates, nthreads=None):
        assert candidates

        root, collapsable = self._select_candidates(candidates)
        ncollapsable = len(collapsable)

        if self._is_offloadable(root) and \
           all(i.is_Affine for i in [root] + collapsable) and \
           self.par_tile:
            # TODO: still unable to exploit multiple par-tiles (one per nest)
            # This will require unconditionally applying blocking, and then infer
            # the tile clause shape from the BlockDimensions' step
            tile = self.par_tile[0]
            assert isinstance(tile, tuple)
            nremainder = (ncollapsable + 1) - len(tile)
            if nremainder >= 0:
                tile += (tile[-1],)*nremainder
            else:
                tile = tile[:ncollapsable + 1]

            body = self.DeviceIteration(gpu_fit=self.gpu_fit, tile=tile, **root.args)
            partree = ParallelTree([], body, nthreads=nthreads)

            return root, partree
        else:
            return super()._make_partree(candidates, nthreads)


class DevicePointerFetch(List):

    def __init__(self, dp, **kwargs):
        hp = dp.mapped.indexed

        tdp = Symbol(name="%s_cpy" % hp.name, dtype=np.uint64)
        init = DummyExpr(tdp, 0, init=True),

        header = c.Pragma("acc serial present(%s) copyout(%s)" % (hp, tdp))
        body = DummyExpr(tdp, cast_mapper[tdp.dtype](hp))
        dpf = Block(header=header, body=body)

        cast = DummyExpr(dp, cast_mapper[(dp.dtype, '*')](tdp), init=True)

        body = [init, dpf, cast]

        super().__init__(body=body)


class DeviceAccDataManager(PragmaDeviceAwareDataManager):

    lang = AccBB

    @iet_pass
    def place_devptr(self, iet, **kwargs):
        """
        Transform `iet` such that device pointers are used in DeviceCalls.

        OpenACC provides multiple mechanisms to this purpose, including the
        `acc_deviceptr(f)` routine and the `host_data use_device(f)` pragma.
        However, none of these will work with `nvc` (until at least v22.3) due
        to a known bug -- see this thread for more info:

            https://forums.developer.nvidia.com/t/acc-deviceptr-does-not-work-in-\
                openacc-code-dynamically-loaded-from-a-shared-library/211599

        Basically, the issue crops up when OpenACC code is part of a shared library
        that is dlopen’d by an executable that is not linked against the `nvc`'s
        OpenACC runtime library. That's our case, since our executable is Python.

        The following work around does the trick:

          .. code-block:: c

            size_t d_f;
            #pragma acc serial present(f) copyout(d_f)
            {
               d_f = (size_t) f;
            }

        This method creates an IET along the lines of the code snippet above.
        """
        mapper = {}
        subs = {}
        for n in FindNodes(DeviceCall).visit(iet):
            for i, t in zip(n.arguments, n.types):
                try:
                    f = i.function
                except AttributeError:
                    continue

                if f.is_Array and f._mem_mapped and t is DPtr:
                    subs[i] = DevicePointer(
                        name="%s_dev" % f.name, dtype=f.dtype, mapped=f
                    )

            mapper[n] = Uxreplace(subs).visit(n)

        if mapper:
            iet = Transformer(mapper).visit(iet)

        args = [i.base for i in subs.values()]

        if not isinstance(iet, EntryFunction):
            return iet, {'args': args}

        nested = [i for i in FindSymbols('basics').visit(iet)
                  if isinstance(i, DevicePointer)]
        dpfs = [DevicePointerFetch(dp) for dp in filter_ordered(args + nested)]

        iet = iet._rebuild(body=iet.body._rebuild(maps=iet.body.maps + tuple(dpfs)))

        return iet, {}


class AccOrchestrator(Orchestrator):
    lang = AccBB

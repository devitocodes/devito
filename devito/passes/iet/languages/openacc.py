import numpy as np

from devito.arch import AMDGPUX, NVIDIAX
from devito.ir import (Call, DeviceCall, DummyExpr, EntryFunction, List, Block,
                       ParallelTree, Pragma, Return, FindSymbols, make_callable)
from devito.passes import needs_transfer, is_on_device
from devito.passes.iet.definitions import DeviceAwareDataManager
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.parpragma import (PragmaDeviceAwareTransformer, PragmaLangBB,
                                         PragmaIteration, PragmaTransfer)
from devito.passes.iet.languages.CXX import CXXBB, CXXPrinter
from devito.passes.iet.languages.openmp import OmpRegion, OmpIteration
from devito.symbolics import FieldFromPointer, Macro, cast
from devito.tools import filter_ordered, UnboundTuple
from devito.types import Symbol

__all__ = ['DeviceAccizer', 'DeviceAccDataManager', 'AccOrchestrator']


class DeviceAccIteration(PragmaIteration):

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'acc parallel loop'

    @classmethod
    def _make_clauses(cls, ncollapsed=0, reduction=None, tile=None, **kwargs):
        clauses = []

        if tile:
            stile = [str(tile[i]) for i in range(ncollapsed)]
            clauses.append('tile(%s)' % ','.join(stile))
        elif ncollapsed > 1:
            clauses.append('collapse(%d)' % ncollapsed)

        if reduction:
            clauses.append(cls._make_clause_reduction_from_imask(reduction))

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


class AccBB(PragmaLangBB):

    BackendCall = DeviceCall

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
        'atomic':
            Pragma('acc atomic update'),
        'map-enter-to': lambda f, imask:
            PragmaTransfer('acc enter data copyin(%s%s)', f, imask=imask),
        'map-enter-to-async': lambda f, imask, a:
            PragmaTransfer('acc enter data copyin(%s%s) async(%s)',
                           f, imask=imask, arguments=a),
        'map-enter-alloc': lambda f, imask:
            PragmaTransfer('acc enter data create(%s%s)', f, imask=imask),
        'map-present': lambda f, imask:
            PragmaTransfer('acc data present(%s%s)', f, imask=imask),
        'map-serial-present': lambda i, j:
            Pragma('acc serial present(%s) copyout(%s)', arguments=(i, j)),
        'map-wait': lambda i:
            Pragma('acc wait(%s)', arguments=i),
        'map-update': lambda f, imask:
            PragmaTransfer('acc exit data copyout(%s%s)', f, imask=imask),
        'map-update-host': lambda f, imask:
            PragmaTransfer('acc update self(%s%s)', f, imask=imask),
        'map-update-host-async': lambda f, imask, a:
            PragmaTransfer('acc update self(%s%s) async(%s)',
                           f, imask=imask, arguments=a),
        'map-update-device': lambda f, imask:
            PragmaTransfer('acc update device(%s%s)', f, imask=imask),
        'map-update-device-async': lambda f, imask, a:
            PragmaTransfer('acc update device(%s%s) async(%s)',
                           f, imask=imask, arguments=a),
        'map-release': lambda f, imask:
            PragmaTransfer('acc exit data delete(%s%s)', f, imask=imask),
        'map-release-if': lambda f, imask, a:
            PragmaTransfer('acc exit data delete(%s%s) if(%s)',
                           f, imask=imask, arguments=a),
        'map-exit-delete': lambda f, imask:
            PragmaTransfer('acc exit data delete(%s%s)', f, imask=imask),
        'map-exit-delete-if': lambda f, imask, a:
            PragmaTransfer('acc exit data delete(%s%s) if(%s)',
                           f, imask=imask, arguments=a),
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

    mapper.update(CXXBB.mapper)

    Region = OmpRegion
    HostIteration = OmpIteration  # Host parallelism still goes via OpenMP
    DeviceIteration = DeviceAccIteration

    @classmethod
    def _map_to_wait(cls, f, imask=None, qid=None):
        return List(body=[
            cls.mapper['map-enter-to-async'](f, imask, qid),
            cls.mapper['map-wait'](qid)
        ])

    @classmethod
    def _map_present(cls, f, imask=None):
        return cls.mapper['map-present'](f, imask)

    @classmethod
    def _map_wait(cls, qid=None):
        return cls.mapper['map-wait'](qid)

    @classmethod
    def _map_delete(cls, f, imask=None, devicerm=None):
        if devicerm:
            return cls.mapper['map-exit-delete-if'](f, imask, devicerm)
        else:
            return cls.mapper['map-exit-delete'](f, imask)

    @classmethod
    def _map_update_host_async(cls, f, imask=None, qid=None):
        return cls.mapper['map-update-host-async'](f, imask, qid)

    @classmethod
    def _map_update_device_async(cls, f, imask=None, qid=None):
        return cls.mapper['map-update-device-async'](f, imask, qid)


class DeviceAccizer(PragmaDeviceAwareTransformer):

    langbb = AccBB

    def _make_partree(self, candidates, nthreads=None):
        assert candidates

        root, collapsable = self._select_candidates(candidates)
        ncollapsable = len(collapsable) + 1

        if self._is_offloadable(root) and \
           all(i.is_Affine for i in [root] + collapsable) and \
           self.par_tile:
            tile = self.par_tile.nextitem()
            assert isinstance(tile, UnboundTuple)

            body = self.DeviceIteration(gpu_fit=self.gpu_fit, tile=tile,
                                        ncollapsed=ncollapsable, **root.args)
            partree = ParallelTree([], body, nthreads=nthreads)

            return root, partree
        else:
            return super()._make_partree(candidates, nthreads)


class DeviceAccDataManager(DeviceAwareDataManager):

    langbb = AccBB

    @iet_pass
    def place_devptr(self, iet, **kwargs):
        """
        Transform `iet` such that device pointers are used in DeviceCalls.

        OpenACC provides multiple mechanisms to this purpose, including the
        `acc_deviceptr(f)` routine and the `host_data use_device(f)` pragma.
        However, none of these will work with `nvc` when compiling shared
        libraries (until at least v22.3) due to a known bug -- see this thread
        for more info:

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
        if not isinstance(iet, EntryFunction):
            return iet, {}

        symbols = FindSymbols('basics').visit(iet)
        functions = [f for f in symbols if needs_transfer(f, self.gpu_fit)]

        efuncs = []
        calls = []
        for f in functions:
            dmap = f.dmap
            hp = f.indexed

            tdp = Symbol(name="dptr", dtype=np.uint64)
            init = DummyExpr(tdp, 0, init=True)

            dpf = List(body=[
                self.langbb.mapper['map-serial-present'](hp, tdp),
                Block(body=DummyExpr(tdp, cast(tdp.dtype)(hp, reinterpret=True)))
            ])

            ffp = FieldFromPointer(f._C_field_dmap, f._C_symbol)
            ctdp = cast(hp.dtype, '*')(tdp, reinterpret=True)
            castf = DummyExpr(ffp, ctdp)

            ret = Return(ctdp)

            body = List(body=[init, dpf, castf, ret])

            name = self.sregistry.make_name(prefix='map_device_ptr')
            efuncs.append(make_callable(name, body, retval=hp))

            if dmap in symbols:
                calls.append(Call(name, f, retobj=dmap))
            else:
                calls.append(Call(name, f))

        body = iet.body._rebuild(maps=iet.body.maps + tuple(calls))
        iet = iet._rebuild(body=body)

        return iet, {'efuncs': efuncs}


class AccOrchestrator(Orchestrator):
    langbb = AccBB


class AccPrinter(CXXPrinter):
    pass

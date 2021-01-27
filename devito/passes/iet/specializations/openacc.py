from functools import singledispatch

import cgen as c

from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Conditional, EntryFunction, List, LocalExpression,
                           ParallelIteration, FindSymbols)
from devito.mpi.distributed import MPICommObject
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.language import Constructs, DeviceAwarePragmaParallelizer, is_on_device
from devito.symbolics import Byref, CondNe, DefFunction, Macro
from devito.types import DeviceID, Symbol

__all__ = ['DeviceAccizer']


class DeviceAccIteration(ParallelIteration):

    def __init__(self, *args, **kwargs):
        pragmas, kwargs = self._make_header(**kwargs)

        properties = as_tuple(kwargs.pop('properties', None))
        properties += (COLLAPSED(kwargs.get('ncollapse', 1)),)

        self.schedule = kwargs.pop('schedule', None)
        self.parallel = kwargs.pop('parallel', False)
        self.ncollapse = kwargs.pop('ncollapse', None)
        self.chunk_size = kwargs.pop('chunk_size', None)
        self.nthreads = kwargs.pop('nthreads', None)
        self.reduction = kwargs.pop('reduction', None)

        super().__init__(*args, pragmas=pragmas, properties=properties, **kwargs)

    @classmethod
    def _make_header(cls, **kwargs):
        kwargs.pop('pragmas', None)

        construct = cls._make_construct(**kwargs)
        clauses = cls._make_clauses(**kwargs)
        header = c.Pragma(' '.join([construct] + clauses))

        return (header,), kwargs

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'acc parallel loop'

    @classmethod
    def _make_clauses(cls, **kwargs):
        clauses = []

        clauses.append('collapse(%d)' % (ncollapse or 1))

        if chunk_size is not False:
            clauses.append('schedule(%s,%s)' % (schedule or 'dynamic',
                                                chunk_size or 1))

        if nthreads:
            clauses.append('num_threads(%s)' % nthreads)

        if reduction:
            args = []
            for i in reduction:
                if i.is_Indexed:
                    f = i.function
                    bounds = []
                    for k, d in zip(i.indices, f.dimensions):
                        if k.is_Number:
                            bounds.append('[%s]' % k)
                        else:
                            # OpenMP expects a range as input of reduction,
                            # such as reduction(+:f[0:f_vec->size[1]])
                            bounds.append('[0:%s]' % f._C_get_field(FULL, d).size)
                    args.append('%s%s' % (i.name, ''.join(bounds)))
                else:
                    args.append(str(i))
            clauses.append('reduction(+:%s)' % ','.join(args))

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


class DeviceAccizer(DeviceAwarePragmaParallelizer):

    lang = Constructs([
        ('atomic', c.Pragma('acc atomic update')),
        ('map-enter-to', lambda i, j:
            c.Pragma('acc enter data copyin(%s%s)' % (i, j))),
        ('map-enter-to-wait', lambda i, j, k:
            (c.Pragma('acc enter data copyin(%s%s) async(%s)' % (i, j, k)),
             c.Pragma('acc wait(%s)' % k))),
        ('map-enter-alloc', lambda i, j:
            c.Pragma('acc enter data create(%s%s)' % (i, j))),
        ('map-present', lambda i, j:
            c.Pragma('acc data present(%s%s)' % (i, j))),
        ('map-update', lambda i, j:
            c.Pragma('acc exit data copyout(%s%s)' % (i, j))),
        ('map-update-host', lambda i, j:
            c.Pragma('acc update self(%s%s)' % (i, j))),
        ('map-update-wait-host', lambda i, j, k:
            (c.Pragma('acc update self(%s%s) async(%s)' % (i, j, k)),
             c.Pragma('acc wait(%s)' % k))),
        ('map-update-device', lambda i, j:
            c.Pragma('acc update device(%s%s)' % (i, j))),
        ('map-update-wait-device', lambda i, j, k:
            (c.Pragma('acc update device(%s%s) async(%s)' % (i, j, k)),
             c.Pragma('acc wait(%s)' % k))),
        ('map-release', lambda i, j, k:
            c.Pragma('acc exit data delete(%s%s)%s' % (i, j, k))),
        ('map-exit-delete', lambda i, j, k:
            c.Pragma('acc exit data delete(%s%s)%s' % (i, j, k))),
        ('header', 'openacc.h')
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

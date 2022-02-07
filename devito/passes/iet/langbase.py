from functools import singledispatch
from abc import ABC

import cgen as c

from devito.data import FULL
from devito.ir import (BlankLine, DummyExpr, Call, Conditional, Expression,
                       List, Prodder, ParallelIteration, ParallelBlock,
                       PointerCast, EntryFunction, ThreadFunction, FindNodes,
                       FindSymbols)
from devito.mpi.distributed import MPICommObject
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.misc import is_on_device
from devito.symbolics import Byref, CondNe
from devito.tools import as_list, prod
from devito.types import Symbol, Wildcard

__all__ = ['LangBB', 'LangTransformer']


class LangMeta(type):

    """
    Metaclass for class-level mappers.
    """

    mapper = {}

    def __getitem__(self, k):
        if k not in self.mapper:
            raise NotImplementedError("Missing required mapping for `%s`" % k)
        return self.mapper[k]


class LangBB(object, metaclass=LangMeta):

    """
    Abstract base class for Language Building Blocks.
    """

    # NOTE: a subclass may want to override the values below, which represent
    # IET node types used in the various lowering and/or transformation passes
    Region = ParallelBlock
    HostIteration = ParallelIteration
    DeviceIteration = ParallelIteration
    Prodder = Prodder
    PointerCast = PointerCast

    @classmethod
    def _map_data(cls, f):
        if f.is_Array:
            return f.symbolic_shape
        else:
            return tuple(f._C_get_field(FULL, d).size for d in f.dimensions)

    @classmethod
    def _make_symbolic_sections_from_imask(cls, f, imask):
        datasize = cls._map_data(f)
        if imask is None:
            imask = [FULL]*len(datasize)

        sections = []
        for i, j in zip(imask, datasize):
            if i is FULL:
                start, size = 0, j
            else:
                try:
                    start, size = i
                except TypeError:
                    start, size = i, 1
            sections.append((start, size))

        # Unroll (or "flatten") the remaining Dimensions not captured by `imask`
        if len(imask) < len(datasize):
            try:
                start, size = sections.pop(-1)
            except IndexError:
                start, size = (0, 1)
            remainder_size = prod(datasize[len(imask):])
            # The reason we may see a Wildcard is detailed in the `linearize_transfer`
            # pass, take a look there for more info. Basically, a Wildcard here means
            # that the symbol `start` is actually a temporary whose value already
            # represents the unrolled size
            if not isinstance(start, Wildcard):
                start *= remainder_size
            size *= remainder_size
            sections.append((start, size))

        return sections

    @classmethod
    def _map_to(cls, f, imask=None, queueid=None):
        """
        Allocate and copy Function from host to device memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_to_wait(cls, f, imask=None, queueid=None):
        """
        Allocate and copy Function from host to device memory and explicitly wait.
        """
        raise NotImplementedError

    @classmethod
    def _map_alloc(cls, f, imask=None):
        """
        Allocate Function in device memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_present(cls, f, imask=None):
        """
        Explicitly flag Function as present in device memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_wait(cls, queueid=None):
        """
        Explicitly wait on event.
        """
        raise NotImplementedError

    @classmethod
    def _map_update(cls, f, imask=None):
        """
        Copyi Function from device to host memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_update_host(cls, f, imask=None, queueid=None):
        """
        Copy Function from device to host memory (alternative to _map_update).
        """
        raise NotImplementedError

    @classmethod
    def _map_update_host_async(cls, f, imask=None, queueid=None):
        """
        Asynchronously copy Function from device to host memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_update_device(cls, f, imask=None, queueid=None):
        """
        Copy Function from host to device memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_update_device_async(cls, f, imask=None, queueid=None):
        """
        Asynchronously copy Function from host to device memory and explicitly wait.
        """
        raise NotImplementedError

    @classmethod
    def _map_release(cls, f, imask=None, devicerm=None):
        """
        Release device pointer to a Function.
        """
        raise NotImplementedError

    @classmethod
    def _map_delete(cls, f, imask=None, devicerm=None):
        """
        Delete Function from device memory.
        """
        raise NotImplementedError


class LangTransformer(ABC):

    """
    Abstract base class defining a series of methods capable of specializing
    an IET for a certain target language (e.g., C, C+OpenMP).
    """

    lang = LangBB
    """
    The constructs of the target language. To be specialized by a subclass.
    """

    def __init__(self, key, sregistry, platform):
        """
        Parameters
        ----------
        key : callable, optional
            Return True if an Iteration can and should be parallelized, False otherwise.
        sregistry : SymbolRegistry
            The symbol registry, to access the symbols appearing in an IET.
        platform : Platform
            The underlying platform.
        """
        if key is not None:
            self.key = key
        else:
            self.key = lambda i: False
        self.sregistry = sregistry
        self.platform = platform

    @iet_pass
    def make_parallel(self, iet):
        """
        An `iet_pass` which transforms an IET for shared-memory parallelism.
        """
        return iet, {}

    @iet_pass
    def make_simd(self, iet):
        """
        An `iet_pass` which transforms an IET for SIMD parallelism.
        """
        return iet, {}

    @iet_pass
    def initialize(self, iet):
        """
        An `iet_pass` which transforms an IET such that the target language
        runtime is initialized.
        """
        return iet, {}

    @property
    def Region(self):
        return self.lang.Region

    @property
    def HostIteration(self):
        return self.lang.HostIteration

    @property
    def DeviceIteration(self):
        return self.lang.DeviceIteration

    @property
    def Prodder(self):
        return self.lang.Prodder


class DeviceAwareMixin(object):

    @property
    def deviceid(self):
        return self.sregistry.deviceid

    @iet_pass
    def initialize(self, iet):
        """
        An `iet_pass` which transforms an IET such that the target language
        runtime is initialized.

        The initialization follows a pattern which is applicable to virtually
        any target language:

            1. Calling the init function (e.g., `acc_init(...)` for OpenACC)
            2. Assignment of the target device to a host thread or an MPI rank
            3. Introduction of user-level symbols (e.g., `deviceid` to allow
               users to select a specific device)

        Despite not all of these are applicable to all target languages, there
        is sufficient reuse to implement the logic as a single method.
        """

        @singledispatch
        def _initialize(iet):
            return iet, {}

        @_initialize.register(EntryFunction)
        def _(iet):
            assert iet.body.is_CallableBody

            # TODO: we need to pick the rank from `comm_shm`, not `comm`,
            # so that we have nranks == ngpus (as long as the user has launched
            # the right number of MPI processes per node given the available
            # number of GPUs per node)

            objcomm = None
            for i in iet.parameters:
                if isinstance(i, MPICommObject):
                    objcomm = i
                    break

            devicetype = as_list(self.lang[self.platform])
            deviceid = self.deviceid

            try:
                lang_init = [self.lang['init'](devicetype)]
            except TypeError:
                # Not all target languages need to be explicitly initialized
                lang_init = []

            if objcomm is not None:
                rank = Symbol(name='rank')
                rank_decl = DummyExpr(rank, 0)
                rank_init = Call('MPI_Comm_rank', [objcomm, Byref(rank)])

                ngpus = Symbol(name='ngpus')
                call = self.lang['num-devices'](devicetype)
                ngpus_init = DummyExpr(ngpus, call)

                osdd_then = self.lang['set-device']([deviceid] + devicetype)
                osdd_else = self.lang['set-device']([rank % ngpus] + devicetype)

                body = lang_init + [Conditional(
                    CondNe(deviceid, -1),
                    osdd_then,
                    List(body=[rank_decl, rank_init, ngpus_init, osdd_else]),
                )]

                header = c.Comment('Begin of %s+MPI setup' % self.lang['name'])
                footer = c.Comment('End of %s+MPI setup' % self.lang['name'])
            else:
                body = lang_init + [Conditional(
                    CondNe(deviceid, -1),
                    self.lang['set-device']([deviceid] + devicetype)
                )]

                header = c.Comment('Begin of %s setup' % self.lang['name'])
                footer = c.Comment('End of %s setup' % self.lang['name'])

            init = List(header=header, body=body, footer=footer)
            iet = iet._rebuild(body=iet.body._rebuild(init=init))

            return iet, {'args': deviceid}

        @_initialize.register(ThreadFunction)
        def _(iet):
            devicetype = as_list(self.lang[self.platform])
            deviceid = self.deviceid

            init = Conditional(
                CondNe(deviceid, -1),
                self.lang['set-device']([deviceid] + devicetype)
            )
            body = iet.body._rebuild(body=(init, BlankLine) + iet.body.body)
            iet = iet._rebuild(body=body)

            return iet, {}

        return _initialize(iet)

    def _is_offloadable(self, iet):
        """
        True if the IET computation is offloadable to device, False otherwise.
        """
        expressions = FindNodes(Expression).visit(iet)
        if any(not is_on_device(e.write, self.gpu_fit) for e in expressions):
            return False

        functions = FindSymbols().visit(iet)
        buffers = [f for f in functions if f.is_Array and f._mem_mapped]
        hostfuncs = [f for f in functions if not is_on_device(f, self.gpu_fit)]
        return not (buffers and hostfuncs)

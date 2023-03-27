from functools import singledispatch
from abc import ABC

import cgen as c

from devito.data import FULL
from devito.ir import (DummyExpr, Call, Conditional, Expression, List, Prodder,
                       ParallelIteration, ParallelBlock, PointerCast, EntryFunction,
                       AsyncCallable, FindNodes, FindSymbols)
from devito.mpi.distributed import MPICommObject
from devito.passes import is_on_device
from devito.passes.iet.engine import iet_pass
from devito.symbolics import Byref, CondNe, SizeOf
from devito.tools import as_list, prod
from devito.types import Symbol, QueueID, Wildcard

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

    # NOTE: a subclass may want to override the attributes below to customize
    # code generation
    BackendCall = Call
    Region = ParallelBlock
    HostIteration = ParallelIteration
    DeviceIteration = ParallelIteration
    Prodder = Prodder
    PointerCast = PointerCast

    AsyncQueue = QueueID
    """
    An object used by the language to enqueue asynchronous operations.
    """

    @classmethod
    def _get_num_devices(cls):
        """
        Get the number of accessible devices.
        """
        raise NotImplementedError

    @classmethod
    def _map_to(cls, f, imask=None, qid=None):
        """
        Allocate and copy Function from host to device memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_to_wait(cls, f, imask=None, qid=None):
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
    def _map_wait(cls, qid=None):
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
    def _map_update_host(cls, f, imask=None, qid=None):
        """
        Copy Function from device to host memory (alternative to _map_update).
        """
        raise NotImplementedError

    @classmethod
    def _map_update_host_async(cls, f, imask=None, qid=None):
        """
        Asynchronously copy Function from device to host memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_update_device(cls, f, imask=None, qid=None):
        """
        Copy Function from host to device memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_update_device_async(cls, f, imask=None, qid=None):
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

    def __init__(self, key, sregistry, platform, compiler):
        """
        Parameters
        ----------
        key : callable, optional
            Return True if an Iteration can and should be parallelized, False otherwise.
        sregistry : SymbolRegistry
            The symbol registry, to access the symbols appearing in an IET.
        platform : Platform
            The underlying platform.
        compiler : Compiler
            The underlying JIT compiler.
        """
        if key is not None:
            self.key = key
        else:
            self.key = lambda i: False
        self.sregistry = sregistry
        self.platform = platform
        self.compiler = compiler

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
    def initialize(self, iet, options=None):
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
    def initialize(self, iet, options=None):
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
            if objcomm is None and options['mpi']:
                # Time to inject `objcomm`. If it's not here, it simply means
                # there's no halo exchanges in the Operator, but we now need it
                # nonetheless to perform the rank-GPU assignment
                for i in iet.parameters:
                    try:
                        objcomm = i.grid.distributor._obj_comm
                        break
                    except AttributeError:
                        pass
                assert objcomm is not None

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

                ngpus, call_ngpus = self.lang._get_num_devices(self.platform)

                osdd_then = self.lang['set-device']([deviceid] + devicetype)
                osdd_else = self.lang['set-device']([rank % ngpus] + devicetype)

                body = lang_init + [Conditional(
                    CondNe(deviceid, -1),
                    osdd_then,
                    List(body=[rank_decl, rank_init, call_ngpus, osdd_else]),
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

            return iet, {}

        @_initialize.register(AsyncCallable)
        def _(iet):
            devicetype = as_list(self.lang[self.platform])
            deviceid = self.deviceid

            init = Conditional(
                CondNe(deviceid, -1),
                self.lang['set-device']([deviceid] + devicetype)
            )
            iet = iet._rebuild(body=iet.body._rebuild(init=init))

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


class Sections(tuple):

    def __new__(cls, function, *args):
        obj = super().__new__(cls, args)
        obj.function = function

        return obj

    @property
    def size(self):
        return prod(s for _, s in self)

    @property
    def nbytes(self):
        return self.size*SizeOf(self.function.indexed._C_typedata)


def make_sections_from_imask(f, imask=None):
    if imask is None:
        imask = [FULL]*f.ndim

    datashape = infer_transfer_datashape(f, imask)

    sections = []
    for i, j in zip(imask, datashape):
        if i is FULL:
            start, size = 0, j
        else:
            try:
                start, size = i
            except TypeError:
                start, size = i, 1
        sections.append((start, size))

    # Unroll (or "flatten") the remaining Dimensions not captured by `imask`
    if len(imask) < len(datashape):
        try:
            start, size = sections.pop(-1)
        except IndexError:
            start, size = (0, 1)
        remainder_size = prod(datashape[len(imask):])
        # The reason we may see a Wildcard is detailed in the `linearize_transfer`
        # pass, take a look there for more info. Basically, a Wildcard here means
        # that the symbol `start` is actually a temporary whose value already
        # represents the unrolled size
        if not isinstance(start, Wildcard):
            start *= remainder_size
        size *= remainder_size
        sections.append((start, size))

    return Sections(f, *sections)


@singledispatch
def infer_transfer_datashape(f, *args):
    """
    Return the best shape to efficiently transfer `f` between the host
    and a device.

    First of all, we observe that the minimum shape is not necessarily the
    best shape, because data contiguity plays a role.

    So even in the simplest case, we transfer *both* the DOMAIN and the
    HALO region. Consider the following:

      .. code-block:: C

        for (int x = x_m; x <= x_M; x += 1)
          for (int y = y_m; y <= y_M; y += 1)
            usaveb0[0][x + 1][y + 1] = u[t1][x + 1][y + 1];

    Here the minimum shape to transfer `usaveb0` would be
    `(1, x_M - x_m + 1, y_M - y_m + 1) == (1, x_size, y_size)`, but
    we would rather transfer the contiguous chunk that also includes the
    HALO region, namely `(1, x_size + 2, y_size + 2)`.

    Likewise, in the case of SubDomains/SubDimensions:

      .. code-block:: C

        for (int xi = x_m + xi_ltkn; xi <= x_M - xi_rtkn; xi += 1)
          for (int yi = y_m + yi_ltkn; yi <= y_M - yi_rtkn; yi += 1)
            usaveb0[0][xi + 1][yi + 1] = u[t1][xi + 1][yi + 1];

    We will transfer `(1, x_size + 2, y_size + 2)`.

    In the future, this behaviour may change, or be made more sophisticated.
    Note that any departure from this simple heuristic will require non trivial
    additions to the compilation toolchain. For example, take the SubDomain
    example above. If we wanted to transfer the minimum shape, that is
    `(1, x_M - x_m - xi_ltkn - xi_rtkn + 1, y_M - y_m - yi_ltkn - yi_rtkn + 1)`,
    we would need to track both the iteration space of the computation and
    the write-to offsets (e.g., `xi + 1` and `yi + 1` in `usaveb0[0][xi + 1][yi + 1]`)
    because clearly we would need to transfer the right amount starting at the
    right offset.

    Finally, we use the single-dispatch paradigm so that this behaviour can
    be customized via external plugins.
    """
    return f.symbolic_shape

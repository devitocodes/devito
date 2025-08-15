from functools import singledispatch
from itertools import takewhile
from abc import ABC

import cgen as c

from devito.data import FULL
from devito.ir import (DummyExpr, Call, Conditional, Expression, List, Prodder,
                       ParallelIteration, ParallelBlock, PointerCast, EntryFunction,
                       AsyncCallable, FindNodes, FindSymbols, IsPerfectIteration)
from devito.mpi.distributed import MPICommObject
from devito.passes import is_on_device
from devito.passes.iet.engine import iet_pass
from devito.symbolics import Byref, CondNe, SizeOf
from devito.tools import as_list, is_integer, prod
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

    def get(self, k, v=None):
        return self.mapper.get(k, v)


class LangBB(metaclass=LangMeta):

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

    langbb = LangBB
    """
    The constructs of the target language. To be specialized by a subclass.
    """

    def __init__(self, key, sregistry, platform, compiler):
        """
        Parameters
        ----------
        key : callable, optional
            Return True if an Iteration can and should be parallelized,
            False otherwise.
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
        return self.langbb.Region

    @property
    def HostIteration(self):
        return self.langbb.HostIteration

    @property
    def DeviceIteration(self):
        return self.langbb.DeviceIteration

    @property
    def Prodder(self):
        return self.langbb.Prodder


class ShmTransformer(LangTransformer):

    """
    Abstract base class for LangTransformers that want to emit
    shared-memory-parallel IETs for CPUs.
    """

    def __init__(self, key, sregistry, options, platform, compiler):
        """
        Parameters
        ----------
        key : callable, optional
            Return True if an Iteration can and should be parallelized,
            False otherwise.
        sregistry : SymbolRegistry
            The symbol registry, to access the symbols appearing in an IET.
        options : dict
             The optimization options.
             Accepted: ['par-collapse-ncores', 'par-collapse-work',
             'par-chunk-nonaffine', 'par-dynamic-work', 'par-nested']
             * 'par-collapse-ncores': use a collapse clause if the number of
               available physical cores is greater than this threshold.
             * 'par-collapse-work': use a collapse clause if the trip count of the
               collapsable Iterations is statically known to exceed this threshold.
             * 'par-chunk-nonaffine': coefficient to adjust the chunk size in
               non-affine parallel Iterations.
             * 'par-dynamic-work': use dynamic scheduling if the operation count per
               iteration exceeds this threshold. Otherwise, use static scheduling.
             * 'par-nested': nested parallelism if the number of hyperthreads
               per core is greater than this threshold.
        platform : Platform
            The underlying platform.
        compiler : Compiler
            The underlying JIT compiler.
        """
        super().__init__(key, sregistry, platform, compiler)

        self.collapse_ncores = options['par-collapse-ncores']
        self.collapse_work = options['par-collapse-work']
        self.chunk_nonaffine = options['par-chunk-nonaffine']
        self.dynamic_work = options['par-dynamic-work']
        self.nested = options['par-nested']

    @property
    def ncores(self):
        return self.platform.cores_physical

    @property
    def nhyperthreads(self):
        return self.platform.threads_per_core

    @property
    def nthreads(self):
        return self.sregistry.nthreads

    @property
    def nthreads_nested(self):
        return self.sregistry.nthreads_nested

    @property
    def nthreads_nonaffine(self):
        return self.sregistry.nthreads_nonaffine

    @property
    def threadid(self):
        return self.sregistry.threadid

    def _score_candidate(self, n0, root, collapsable=()):
        """
        The score of a collapsable nest depends on the number of fully-parallel
        Iterations and their position in the nest (the outer, the better).
        """
        nest = [root] + list(collapsable)
        n = len(nest)

        # Number of fully-parallel collapsable Iterations
        key = lambda i: i.is_ParallelNoAtomic
        fp_iters = list(takewhile(key, nest))
        n_fp_iters = len(fp_iters)

        # Number of parallel-if-atomic collapsable Iterations
        key = lambda i: i.is_ParallelAtomic
        pia_iters = list(takewhile(key, nest))
        n_pia_iters = len(pia_iters)

        # Prioritize the Dimensions that are more likely to define larger
        # iteration spaces
        key = lambda d: (not d.is_Derived or
                         (d.is_Custom and not is_integer(d.symbolic_size)) or
                         (d.is_Block and d._depth == 1))

        fpdims = [i.dim for i in fp_iters]
        n_fp_iters_large = len([d for d in fpdims if key(d)])

        piadims = [i.dim for i in pia_iters]
        n_pia_iters_large = len([d for d in piadims if key(d)])

        return (
            int(n_fp_iters == n),  # Fully-parallel nest
            n_fp_iters_large,
            n_fp_iters,
            n_pia_iters_large,
            n_pia_iters,
            -(n0 + 1),  # The outer, the better
            n,
        )

    def _select_candidates(self, candidates):
        assert candidates

        if self.ncores < self.collapse_ncores:
            return candidates[0], []

        mapper = {}
        for n0, root in enumerate(candidates):

            # Score `root` in isolation
            mapper[(root, ())] = self._score_candidate(n0, root)

            collapsable = []
            for n, i in enumerate(candidates[n0+1:], n0+1):
                # The Iteration nest [root, ..., i] must be perfect
                if not IsPerfectIteration(depth=i).visit(root):
                    break

                # Loops are collapsable only if none of the iteration variables
                # appear in initializer expressions. For example, the following
                # two loops cannot be collapsed
                #
                # for (i = ... )
                #   for (j = i ...)
                #     ...
                #
                # Here, we make sure this won't happen
                if any(j.dim in i.symbolic_min.free_symbols for j in candidates[n0:n]):
                    break

                # Can't collapse SIMD-vectorized Iterations
                if i.is_Vectorized:
                    break

                # Would there be enough work per parallel iteration?
                nested = candidates[n+1:]
                if nested:
                    try:
                        work = prod([int(j.dim.symbolic_size) for j in nested])
                        if work < self.collapse_work:
                            break
                    except TypeError:
                        pass

                collapsable.append(i)

                # Score `root + collapsable`
                v = tuple(collapsable)
                mapper[(root, v)] = self._score_candidate(n0, root, v)

        # Retrieve the candidates with highest score
        root, collapsable = max(mapper, key=mapper.get)

        return root, list(collapsable)


class DeviceAwareMixin:

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

        def _extract_objcomm(iet):
            for i in iet.parameters:
                if isinstance(i, MPICommObject):
                    return i

            # Fallback -- might end up here because the Operator has no
            # halo exchanges, but we now need it nonetheless to perform
            # the rank-GPU assignment
            if options['mpi']:
                for i in iet.parameters:
                    try:
                        return i.grid.distributor._obj_comm
                    except AttributeError:
                        pass

        def _make_setdevice_seq(iet, nodes=()):
            devicetype = as_list(self.langbb[self.platform])
            deviceid = self.deviceid

            return list(nodes) + [Conditional(
                CondNe(deviceid, -1),
                self.langbb['set-device']([deviceid] + devicetype)
            )]

        def _make_setdevice_mpi(iet, objcomm, nodes=()):
            devicetype = as_list(self.langbb[self.platform])
            deviceid = self.deviceid

            rank = Symbol(name='rank')
            rank_decl = DummyExpr(rank, 0)
            rank_init = Call('MPI_Comm_rank', [objcomm, Byref(rank)])

            ngpus, call_ngpus = self.langbb._get_num_devices(self.platform)

            osdd_then = self.langbb['set-device']([deviceid] + devicetype)
            osdd_else = self.langbb['set-device']([rank % ngpus] + devicetype)

            return list(nodes) + [Conditional(
                CondNe(deviceid, -1),
                osdd_then,
                List(body=[rank_decl, rank_init, call_ngpus, osdd_else]),
            )]

        @singledispatch
        def _initialize(iet):
            return iet, {}

        @_initialize.register(EntryFunction)
        def _(iet):
            assert iet.body.is_CallableBody

            devicetype = as_list(self.langbb[self.platform])

            try:
                lang_init = [self.langbb['init'](devicetype)]
            except TypeError:
                # Not all target languages need to be explicitly initialized
                lang_init = []

            objcomm = _extract_objcomm(iet)

            if objcomm is not None:
                body = _make_setdevice_mpi(iet, objcomm, nodes=lang_init)

                header = c.Comment('Beginning of %s+MPI setup' % self.langbb['name'])
                footer = c.Comment('End of %s+MPI setup' % self.langbb['name'])
            else:
                body = _make_setdevice_seq(iet, nodes=lang_init)

                header = c.Comment('Beginning of %s setup' % self.langbb['name'])
                footer = c.Comment('End of %s setup' % self.langbb['name'])

            init = List(header=header, body=body, footer=footer)
            iet = iet._rebuild(body=iet.body._rebuild(init=init))

            return iet, {}

        @_initialize.register(AsyncCallable)
        def _(iet):
            objcomm = _extract_objcomm(iet)
            if objcomm is not None:
                init = _make_setdevice_mpi(iet, objcomm)
            else:
                init = _make_setdevice_seq(iet)

            iet = iet._rebuild(body=iet.body._rebuild(init=init))

            return iet, {}

        return _initialize(iet)

    def _device_pointers(self, iet):
        functions = FindSymbols().visit(iet)
        devfuncs = [f for f in functions if f.is_Array and f._mem_local]
        return set(devfuncs)

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

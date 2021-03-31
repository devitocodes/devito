from functools import singledispatch
from abc import ABC

import cgen as c

from devito.ir import (DummyEq, Call, Conditional, List, Prodder, ParallelIteration,
                       ParallelBlock, PointerCast, EntryFunction, LocalExpression)
from devito.mpi.distributed import MPICommObject
from devito.passes.iet.engine import iet_pass
from devito.symbolics import Byref, CondNe
from devito.tools import as_list
from devito.types import DeviceID, Symbol

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
    def _map_update(cls, f):
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
    def _map_update_wait_host(cls, f, imask=None, queueid=None):
        """
        Copy Function from device to host memory and explicitly wait.
        """
        raise NotImplementedError

    @classmethod
    def _map_update_device(cls, f, imask=None, queueid=None):
        """
        Copy Function from host to device memory.
        """
        raise NotImplementedError

    @classmethod
    def _map_update_wait_device(cls, f, imask=None, queueid=None):
        """
        Copy Function from host to device memory and explicitly wait.
        """
        raise NotImplementedError

    @classmethod
    def _map_release(cls, f, devicerm=None):
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

            try:
                lang_init = [self.lang['init'](devicetype)]
            except TypeError:
                # Not all target languages need to be explicitly initialized
                lang_init = []

            deviceid = DeviceID()
            if objcomm is not None:
                rank = Symbol(name='rank')
                rank_decl = LocalExpression(DummyEq(rank, 0))
                rank_init = Call('MPI_Comm_rank', [objcomm, Byref(rank)])

                ngpus = Symbol(name='ngpus')
                call = self.lang['num-devices'](devicetype)
                ngpus_init = LocalExpression(DummyEq(ngpus, call))

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

            init = List(header=header, body=body, footer=(footer, c.Line()))
            iet = iet._rebuild(body=(init,) + iet.body)

            return iet, {'args': deviceid}

        return _initialize(iet)

    @iet_pass
    def make_gpudirect(self, iet):
        """
        An `iet_pass` which transforms an IET modifying all MPI Callables such
        that device pointers are used in place of host pointers, thus exploiting
        GPU-direct communication.
        """
        return iet, {}

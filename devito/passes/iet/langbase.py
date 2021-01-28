from abc import ABC

from devito.ir import ParallelBlock, ParallelIteration, Prodder
from devito.passes.iet.engine import iet_pass

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

    # Note: below dummy values are used, so a subclass should override them

    Region = ParallelBlock
    """
    The IET node type to be used to construct a parallel region.
    """

    HostIteration = ParallelIteration
    """
    The IET node type to be used to construct a host-parallel Iteration.
    """

    DeviceIteration = ParallelIteration
    """
    The IET node type to be used to construct a device-parallel Iteration.
    """

    Prodder = Prodder
    """
    The IET node type to be used to construct asynchronous prodders.
    """

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
        return self.lang.Region

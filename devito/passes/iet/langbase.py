from abc import ABC

from devito.passes.iet.engine import iet_pass

__all__ = ['LangBB', 'LangTransformer']


class LangBB(dict):

    """
    Language Building Blocks.
    """

    def __getitem__(self, k):
        if k not in self:
            raise NotImplementedError("Must implement `lang[%s]`" % k)
        return super().__getitem__(k)


class LangTransformer(ABC):

    """
    Abstract base class defining a series of methods capable of specializing
    an IET for a certain target language (e.g., C, C+OpenMP).
    """

    lang = LangBB()
    """
    The constructs of the target language.
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

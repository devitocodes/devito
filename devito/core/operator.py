from devito.core.autotuning import autotune
from devito.parameters import configuration
from devito.passes import NThreads, NThreadsNested, NThreadsNonaffine
from devito.operator import Operator
from devito.tools import generator

__all__ = ['OperatorCore']


class SymbolRegistry(object):

    """A registry for all the symbols used by a `core` Operator."""

    _symbol_prefix = 'r'

    def __init__(self, nthreads=None, nthreads_nested=None, nthreads_nonaffine=None):
        # {name -> generator()} -- to create unique names for symbols, functions, ...
        self.counters = {}

        # Special symbols
        self.nthreads = nthreads
        self.nthreads_nested = nthreads_nested
        self.nthreads_nonaffine = nthreads_nonaffine

    def make_name(self, prefix=None):
        # By default we're creating a new symbol
        if prefix is None:
            prefix = self._symbol_prefix

        try:
            counter = self.counters[prefix]
        except KeyError:
            counter = self.counters.setdefault(prefix, generator())

        return "%s%d" % (prefix, counter())


class OperatorCore(Operator):

    @classmethod
    def _symbol_registry(cls):
        return SymbolRegistry(nthreads=NThreads(aliases='nthreads0'),
                              nthreads_nested=NThreadsNested(aliases='nthreads1'),
                              nthreads_nonaffine=NThreadsNonaffine(aliases='nthreads2'))

    def _autotune(self, args, setup):
        if setup in [False, 'off']:
            return args
        elif setup is True:
            level, mode = configuration['autotuning']
            level = level or 'basic'
            args, summary = autotune(self, args, level, mode)
        elif isinstance(setup, str):
            _, mode = configuration['autotuning']
            args, summary = autotune(self, args, setup, mode)
        elif isinstance(setup, tuple) and len(setup) == 2:
            level, mode = setup
            if level is False:
                return args
            else:
                args, summary = autotune(self, args, level, mode)
        else:
            raise ValueError("Expected bool, str, or 2-tuple, got `%s` instead"
                             % type(setup))

        # Record the tuned values
        self._state.setdefault('autotuning', []).append(summary)

        return args

    @property
    def nthreads(self):
        nthreads = [i for i in self.input if type(i).__base__ is NThreads]
        if len(nthreads) == 0:
            return 1
        else:
            assert len(nthreads) == 1
            return nthreads.pop()

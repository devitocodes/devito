from devito.core.autotuning import autotune
from devito.parameters import configuration
from devito.passes import NThreads, NThreadsNested, NThreadsNonaffine
from devito.operator import Operator, SymbolRegistry
from devito.types import ThreadDimension

__all__ = ['OperatorCore']


class OperatorCore(Operator):

    @classmethod
    def _symbol_registry(cls):
        nthreads = NThreads(aliases='nthreads0')
        nthreads_nested = NThreadsNested(aliases='nthreads1')
        nthreads_nonaffine = NThreadsNonaffine(aliases='nthreads2')
        threadid = ThreadDimension(nthreads=nthreads)

        return SymbolRegistry(nthreads, nthreads_nested, nthreads_nonaffine, threadid)

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

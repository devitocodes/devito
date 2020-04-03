from devito.core.autotuning import autotune
from devito.ir.support import align_accesses
from devito.parameters import configuration
from devito.passes import NThreads
from devito.operator import Operator

__all__ = ['OperatorCore']


class OperatorCore(Operator):

    @classmethod
    def _specialize_exprs(cls, expressions):
        # Align data accesses to the computational domain
        key = lambda i: i.is_DiscreteFunction
        expressions = [align_accesses(e, key=key) for e in expressions]
        return super(OperatorCore, cls)._specialize_exprs(expressions)

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

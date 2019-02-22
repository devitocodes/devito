from devito.core.autotuning import autotune
from devito.ir.support import align_accesses
from devito.parameters import configuration
from devito.operator import Operator

__all__ = ['OperatorCore']


class OperatorCore(Operator):

    def _specialize_exprs(self, expressions):
        # Align data accesses to the computational domain
        key = lambda i: i.is_DiscreteFunction
        expressions = [align_accesses(e, key=key) for e in expressions]
        return super(OperatorCore, self)._specialize_exprs(expressions)

    def _autotune(self, args, setup):
        if setup is False:
            return args
        elif setup is True:
            level = configuration['autotuning'].level or 'basic'
            args, summary = autotune(self, args, level, configuration['autotuning'].mode)
        elif isinstance(setup, str):
            args, summary = autotune(self, args, setup, configuration['autotuning'].mode)
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

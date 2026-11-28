"""Torch language specializations of the standard CPU operator pipeline."""

from devito.core.cpu import (
    Cpu64AdvOperator, Cpu64CustomOperator, Cpu64FsgOperator, Cpu64NoopOperator
)
from devito.exceptions import InvalidOperator
from devito.ir.iet.torch import TorchCGen
from devito.passes.iet import TorchTarget, linearize
from devito.tools import timed_pass


class TorchOperatorMixin:
    """Reuse CPU optimization passes with the Torch C++ printer and target."""

    _ccode_handler = TorchCGen
    _Target = TorchTarget
    _compiler_options = {'cpp': True, 'std': 'c++20'}

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        if kwargs['options'].get('linearize') is False:
            raise InvalidOperator("Torch C++ code generation requires linearization")
        kwargs['options']['linearize'] = True
        return super()._normalize_kwargs(**kwargs)


class Cpu64NoopTorchOperator(TorchOperatorMixin, Cpu64NoopOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        graph = super()._specialize_iet(graph, **kwargs)
        linearize(graph, **kwargs)
        return graph


class Cpu64AdvTorchOperator(TorchOperatorMixin, Cpu64AdvOperator):
    pass


class Cpu64FsgTorchOperator(TorchOperatorMixin, Cpu64FsgOperator):
    pass


class Cpu64CustomTorchOperator(TorchOperatorMixin, Cpu64CustomOperator):
    pass

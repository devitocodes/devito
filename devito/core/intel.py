from devito.core.cpu import (Cpu64AdvCOperator, Cpu64AdvOmpOperator,
                             Cpu64FsgCOperator, Cpu64FsgOmpOperator)

__all__ = ['Intel64AdvCOperator', 'Intel64AdvOmpOperator', 'Intel64FsgCOperator',
           'Intel64FsgOmpOperator']


Intel64AdvCOperator = Cpu64AdvCOperator
Intel64AdvOmpOperator = Cpu64AdvOmpOperator
Intel64FsgCOperator = Cpu64FsgCOperator
Intel64FsgOmpOperator = Cpu64FsgOmpOperator

from devito.core.cpu import (
    Cpu64AdvCOperator, Cpu64AdvCXXOmpOperator, Cpu64AdvCXXOperator, Cpu64AdvOmpOperator,
    Cpu64FsgCOperator, Cpu64FsgCXXOmpOperator, Cpu64FsgCXXOperator, Cpu64FsgOmpOperator
)

__all__ = [
                             'Intel64AdvCOperator',
                             'Intel64AdvCXXOmpOperator',
                             'Intel64AdvOmpOperator',
                             'Intel64CXXAdvCOperator',
                             'Intel64FsgCOperator',
                             'Intel64FsgCXXOmpOperator',
                             'Intel64FsgCXXOperator',
                             'Intel64FsgOmpOperator',
]


Intel64AdvCOperator = Cpu64AdvCOperator
Intel64AdvOmpOperator = Cpu64AdvOmpOperator
Intel64FsgCOperator = Cpu64FsgCOperator
Intel64FsgOmpOperator = Cpu64FsgOmpOperator
Intel64CXXAdvCOperator = Cpu64AdvCXXOperator
Intel64AdvCXXOmpOperator = Cpu64AdvCXXOmpOperator
Intel64FsgCXXOperator = Cpu64FsgCXXOperator
Intel64FsgCXXOmpOperator = Cpu64FsgCXXOmpOperator

from devito.core.cpu import (Cpu64AdvCOperator, Cpu64AdvOmpOperator,
                             Cpu64FsgCOperator, Cpu64FsgOmpOperator,
                             Cpu64AdvCXXOperator, Cpu64CXXAdvOmpOperator,
                             Cpu64CXXFsgCOperator, Cpu64CXXFsgOmpOperator)

__all__ = ['Intel64AdvCOperator', 'Intel64AdvOmpOperator', 'Intel64FsgCOperator',
           'Intel64FsgOmpOperator', 'Intel64CXXAdvCOperator', 'Intel64CXXAdvOmpOperator',
           'Intel64CXXFsgCOperator', 'Intel64CXXFsgOmpOperator']


Intel64AdvCOperator = Cpu64AdvCOperator
Intel64AdvOmpOperator = Cpu64AdvOmpOperator
Intel64FsgCOperator = Cpu64FsgCOperator
Intel64FsgOmpOperator = Cpu64FsgOmpOperator
Intel64CXXAdvCOperator = Cpu64AdvCXXOperator
Intel64CXXAdvOmpOperator = Cpu64CXXAdvOmpOperator
Intel64CXXFsgCOperator = Cpu64CXXFsgCOperator
Intel64CXXFsgOmpOperator = Cpu64CXXFsgOmpOperator

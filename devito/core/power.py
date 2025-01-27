from devito.core.cpu import (Cpu64AdvCOperator, Cpu64AdvOmpOperator,
                             Cpu64AdvCXXOperator, Cpu64CXXAdvOmpOperator)

__all__ = ['PowerAdvCOperator', 'PowerAdvOmpOperator',
           'PowerCXXAdvCOperator', 'PowerCXXAdvOmpOperator']

PowerAdvCOperator = Cpu64AdvCOperator
PowerAdvOmpOperator = Cpu64AdvOmpOperator
PowerCXXAdvCOperator = Cpu64AdvCXXOperator
PowerCXXAdvOmpOperator = Cpu64CXXAdvOmpOperator

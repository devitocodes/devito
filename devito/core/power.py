from devito.core.cpu import (Cpu64AdvCOperator, Cpu64AdvOmpOperator,
                             Cpu64AdvCXXOperator, Cpu64AdvCXXOmpOperator)

__all__ = ['PowerAdvCOperator', 'PowerAdvOmpOperator',
           'PowerCXXAdvCOperator', 'PowerAdvCXXOmpOperator']

PowerAdvCOperator = Cpu64AdvCOperator
PowerAdvOmpOperator = Cpu64AdvOmpOperator
PowerCXXAdvCOperator = Cpu64AdvCXXOperator
PowerAdvCXXOmpOperator = Cpu64AdvCXXOmpOperator

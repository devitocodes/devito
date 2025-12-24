from devito.core.cpu import (
    Cpu64AdvCOperator, Cpu64AdvCXXOmpOperator, Cpu64AdvCXXOperator, Cpu64AdvOmpOperator
)

__all__ = [
                             'PowerAdvCOperator',
                             'PowerAdvCXXOmpOperator',
                             'PowerAdvOmpOperator',
                             'PowerCXXAdvCOperator',
]

PowerAdvCOperator = Cpu64AdvCOperator
PowerAdvOmpOperator = Cpu64AdvOmpOperator
PowerCXXAdvCOperator = Cpu64AdvCXXOperator
PowerAdvCXXOmpOperator = Cpu64AdvCXXOmpOperator

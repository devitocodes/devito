from devito.core.cpu import (Cpu64AdvOperator, Cpu64AdvCXXOperator,
                             Cpu64AdvCOperator)
from devito.passes.iet import OmpTarget, CXXOmpTarget

__all__ = ['ArmAdvCOperator', 'ArmAdvOmpOperator', 'ArmAdvCXXOperator',
           'ArmAdvCXXOmpOperator']


ArmAdvOperator = Cpu64AdvOperator
ArmAdvCOperator = Cpu64AdvCOperator
ArmAdvCXXOperator = Cpu64AdvCXXOperator


class ArmAdvOmpOperator(ArmAdvCOperator):
    _Target = OmpTarget

    # Avoid nested parallelism on ThunderX2
    PAR_NESTED = 4


class ArmAdvCXXOmpOperator(ArmAdvOmpOperator):
    _Target = CXXOmpTarget
    LINEARIZE = True

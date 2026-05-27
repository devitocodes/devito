from devito.core.cpu import Cpu64AdvCOperator, Cpu64AdvCXXOperator, Cpu64AdvOperator
from devito.passes.iet import CXXOmpTarget, OmpTarget

__all__ = [
    'ArmAdvCOperator',
    'ArmAdvCXXOmpOperator',
    'ArmAdvCXXOperator',
    'ArmAdvOmpOperator',
]


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

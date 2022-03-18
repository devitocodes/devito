from devito.core.cpu import Cpu64AdvOperator
from devito.passes.iet import CTarget, OmpTarget

__all__ = ['ArmAdvCOperator', 'ArmAdvOmpOperator']


class ArmAdvOperator(Cpu64AdvOperator):
    pass


class ArmAdvCOperator(ArmAdvOperator):
    _Target = CTarget


class ArmAdvOmpOperator(ArmAdvOperator):
    _Target = OmpTarget

    # Avoid nested parallelism on ThunderX2
    PAR_NESTED = 4

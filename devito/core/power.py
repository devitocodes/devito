from devito.core.cpu import CPU64Operator, CPU64OpenMPOperator

__all__ = ['PowerOperator', 'PowerOpenMPOperator']

PowerOperator = CPU64Operator
PowerOpenMPOperator = CPU64OpenMPOperator

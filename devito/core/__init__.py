from devito.archinfo import Cpu64, Intel64, Arm, Power, Device
from devito.core.cpu import (CPU64NoopOperator, CPU64Operator, Intel64Operator,
                             Intel64FSGOperator, ArmOperator, PowerOperator,
                             CustomOperator)
from devito.core.gpu_openmp import (DeviceOpenMPNoopOperator, DeviceOpenMPOperator,
                                    DeviceOpenMPCustomOperator)
from devito.core.gpu_openacc import (DeviceOpenACCNoopOperator, DeviceOpenACCOperator,
                                     DeviceOpenACCCustomOperator)
from devito.operator.registry import operator_registry
from devito.parameters import Parameters, add_sub_configuration

core_configuration = Parameters('core')
add_sub_configuration(core_configuration)

# Add core-specific Operators
operator_registry.add(CPU64NoopOperator, Cpu64, 'noop')
operator_registry.add(CPU64Operator, Cpu64, 'advanced')
operator_registry.add(CustomOperator, Cpu64, 'custom')

operator_registry.add(CPU64NoopOperator, Intel64, 'noop')
operator_registry.add(Intel64Operator, Intel64, 'advanced')
operator_registry.add(Intel64FSGOperator, Intel64, 'advanced-fsg')
operator_registry.add(CustomOperator, Intel64, 'custom')

operator_registry.add(CPU64NoopOperator, Arm, 'noop')
operator_registry.add(ArmOperator, Arm, 'advanced')
operator_registry.add(CustomOperator, Arm, 'custom')

operator_registry.add(CPU64NoopOperator, Power, 'noop')
operator_registry.add(PowerOperator, Power, 'advanced')
operator_registry.add(CustomOperator, Power, 'custom')

operator_registry.add(DeviceOpenMPNoopOperator, Device, 'noop')
operator_registry.add(DeviceOpenMPOperator, Device, 'advanced')
operator_registry.add(DeviceOpenMPCustomOperator, Device, 'custom')

operator_registry.add(DeviceOpenACCNoopOperator, Device, 'noop', 'openacc')
operator_registry.add(DeviceOpenACCOperator, Device, 'advanced', 'openacc')
operator_registry.add(DeviceOpenACCCustomOperator, Device, 'custom', 'openacc')

# The following used by backends.backendSelector
from devito.core.operator import OperatorCore as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.caching import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

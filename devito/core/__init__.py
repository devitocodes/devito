from devito.archinfo import Cpu64, Intel64, Arm, Power, Device
from devito.core.cpu import (CPU64NoopOperator, CPU64Operator, CPU64OpenMPOperator,
                             Intel64Operator, Intel64OpenMPOperator, Intel64FSGOperator,
                             Intel64FSGOpenMPOperator, ArmOperator, ArmOpenMPOperator,
                             PowerOperator, PowerOpenMPOperator, CustomOperator)
from devito.core.gpu_openmp import (DeviceOpenMPNoopOperator, DeviceOpenMPOperator,
                                    DeviceOpenMPCustomOperator)
from devito.core.gpu_openacc import (DeviceOpenACCNoopOperator, DeviceOpenACCOperator,
                                     DeviceOpenACCCustomOperator)
from devito.operator.registry import operator_registry
from devito.parameters import Parameters, add_sub_configuration

core_configuration = Parameters('core')
add_sub_configuration(core_configuration)

# Register CPU Operators
operator_registry.add(CustomOperator, Cpu64, 'custom', 'C')
operator_registry.add(CustomOperator, Cpu64, 'custom', 'openmp')

operator_registry.add(CPU64NoopOperator, Cpu64, 'noop', 'C')
operator_registry.add(CPU64NoopOperator, Cpu64, 'noop', 'openmp')

operator_registry.add(CPU64Operator, Cpu64, 'advanced', 'C')
operator_registry.add(CPU64OpenMPOperator, Cpu64, 'advanced', 'openmp')

operator_registry.add(Intel64Operator, Intel64, 'advanced', 'C')
operator_registry.add(Intel64OpenMPOperator, Intel64, 'advanced', 'openmp')
operator_registry.add(Intel64FSGOperator, Intel64, 'advanced-fsg', 'C')
operator_registry.add(Intel64FSGOpenMPOperator, Intel64, 'advanced-fsg', 'openmp')

operator_registry.add(ArmOperator, Arm, 'advanced', 'C')
operator_registry.add(ArmOpenMPOperator, Arm, 'advanced', 'openmp')

operator_registry.add(PowerOperator, Power, 'advanced', 'C')
operator_registry.add(PowerOpenMPOperator, Power, 'advanced', 'openmp')

# Register Device Operators
operator_registry.add(DeviceOpenMPCustomOperator, Device, 'custom', 'C')
operator_registry.add(DeviceOpenMPCustomOperator, Device, 'custom', 'openmp')
operator_registry.add(DeviceOpenACCCustomOperator, Device, 'custom', 'openacc')

operator_registry.add(DeviceOpenMPNoopOperator, Device, 'noop', 'C')
operator_registry.add(DeviceOpenMPNoopOperator, Device, 'noop', 'openmp')
operator_registry.add(DeviceOpenACCNoopOperator, Device, 'noop', 'openacc')

operator_registry.add(DeviceOpenMPOperator, Device, 'advanced', 'C')
operator_registry.add(DeviceOpenMPOperator, Device, 'advanced', 'openmp')
operator_registry.add(DeviceOpenACCOperator, Device, 'advanced', 'openacc')

# The following used by backends.backendSelector
from devito.core.operator import OperatorCore as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.caching import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

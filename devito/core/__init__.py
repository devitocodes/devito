from devito.arch import Cpu64, Intel64, Arm, Power, Device
from devito.core.cpu import (Cpu64NoopOperator, Cpu64AdvOperator, Cpu64AdvOmpOperator,
                             Cpu64CustomOperator)
from devito.core.intel import (Intel64AdvOperator, Intel64AdvOmpOperator,
                               Intel64FsgOperator, Intel64FsgOmpOperator)
from devito.core.arm import ArmAdvOperator, ArmAdvOmpOperator
from devito.core.power import PowerAdvOperator, PowerAdvOmpOperator
from devito.core.gpu_openmp import (DeviceNoopOmpOperator, DeviceAdvOmpOperator,
                                    DeviceCustomOmpOperator)
from devito.core.gpu_openacc import (DeviceNoopAccOperator, DeviceAdvAccOperator,
                                     DeviceCustomAccOperator)
from devito.operator.registry import operator_registry
from devito.parameters import Parameters, add_sub_configuration

core_configuration = Parameters('core')
add_sub_configuration(core_configuration)

# Register CPU Operators
operator_registry.add(Cpu64CustomOperator, Cpu64, 'custom', 'C')
operator_registry.add(Cpu64CustomOperator, Cpu64, 'custom', 'openmp')

operator_registry.add(Cpu64NoopOperator, Cpu64, 'noop', 'C')
operator_registry.add(Cpu64NoopOperator, Cpu64, 'noop', 'openmp')
#TODO: Add Cpu64NoopOmpOperator

operator_registry.add(Cpu64AdvOperator, Cpu64, 'advanced', 'C')
operator_registry.add(Cpu64AdvOmpOperator, Cpu64, 'advanced', 'openmp')

operator_registry.add(Intel64AdvOperator, Intel64, 'advanced', 'C')
operator_registry.add(Intel64AdvOmpOperator, Intel64, 'advanced', 'openmp')
operator_registry.add(Intel64FsgOperator, Intel64, 'advanced-fsg', 'C')
operator_registry.add(Intel64FsgOmpOperator, Intel64, 'advanced-fsg', 'openmp')

operator_registry.add(ArmAdvOperator, Arm, 'advanced', 'C')
operator_registry.add(ArmAdvOmpOperator, Arm, 'advanced', 'openmp')

operator_registry.add(PowerAdvOperator, Power, 'advanced', 'C')
operator_registry.add(PowerAdvOmpOperator, Power, 'advanced', 'openmp')

# Register Device Operators
operator_registry.add(DeviceCustomOmpOperator, Device, 'custom', 'C')
operator_registry.add(DeviceCustomOmpOperator, Device, 'custom', 'openmp')
operator_registry.add(DeviceCustomAccOperator, Device, 'custom', 'openacc')

operator_registry.add(DeviceNoopOmpOperator, Device, 'noop', 'C')
operator_registry.add(DeviceNoopOmpOperator, Device, 'noop', 'openmp')
operator_registry.add(DeviceNoopAccOperator, Device, 'noop', 'openacc')

operator_registry.add(DeviceAdvOmpOperator, Device, 'advanced', 'C')
operator_registry.add(DeviceAdvOmpOperator, Device, 'advanced', 'openmp')
operator_registry.add(DeviceAdvAccOperator, Device, 'advanced', 'openacc')

# The following used by backends.backendSelector
from devito.core.operator import CoreOperator as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.caching import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

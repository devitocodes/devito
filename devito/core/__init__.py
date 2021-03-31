from devito.arch import Cpu64, Intel64, Arm, Power, Device
from devito.core.cpu import (Cpu64NoopCOperator, Cpu64NoopOmpOperator,
                             Cpu64AdvCOperator, Cpu64AdvOmpOperator,
                             Cpu64FsgCOperator, Cpu64FsgOmpOperator,
                             Cpu64CustomOperator)
from devito.core.intel import (Intel64AdvCOperator, Intel64AdvOmpOperator,
                               Intel64FsgCOperator, Intel64FsgOmpOperator)
from devito.core.arm import ArmAdvCOperator, ArmAdvOmpOperator
from devito.core.power import PowerAdvCOperator, PowerAdvOmpOperator
from devito.core.gpu import (DeviceNoopOmpOperator, DeviceNoopAccOperator,
                             DeviceAdvOmpOperator, DeviceAdvAccOperator,
                             DeviceFsgOmpOperator, DeviceFsgAccOperator,
                             DeviceCustomOmpOperator, DeviceCustomAccOperator)
from devito.operator.registry import operator_registry

# Register CPU Operators
operator_registry.add(Cpu64CustomOperator, Cpu64, 'custom', 'C')
operator_registry.add(Cpu64CustomOperator, Cpu64, 'custom', 'openmp')

operator_registry.add(Cpu64NoopCOperator, Cpu64, 'noop', 'C')
operator_registry.add(Cpu64NoopOmpOperator, Cpu64, 'noop', 'openmp')

operator_registry.add(Cpu64AdvCOperator, Cpu64, 'advanced', 'C')
operator_registry.add(Cpu64AdvOmpOperator, Cpu64, 'advanced', 'openmp')

operator_registry.add(Cpu64FsgCOperator, Cpu64, 'advanced-fsg', 'C')
operator_registry.add(Cpu64FsgOmpOperator, Cpu64, 'advanced-fsg', 'openmp')

operator_registry.add(Intel64AdvCOperator, Intel64, 'advanced', 'C')
operator_registry.add(Intel64AdvOmpOperator, Intel64, 'advanced', 'openmp')
operator_registry.add(Intel64FsgCOperator, Intel64, 'advanced-fsg', 'C')
operator_registry.add(Intel64FsgOmpOperator, Intel64, 'advanced-fsg', 'openmp')

operator_registry.add(ArmAdvCOperator, Arm, 'advanced', 'C')
operator_registry.add(ArmAdvOmpOperator, Arm, 'advanced', 'openmp')

operator_registry.add(PowerAdvCOperator, Power, 'advanced', 'C')
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

operator_registry.add(DeviceFsgOmpOperator, Device, 'advanced-fsg', 'C')
operator_registry.add(DeviceFsgOmpOperator, Device, 'advanced-fsg', 'openmp')
operator_registry.add(DeviceFsgAccOperator, Device, 'advanced-fsg', 'openacc')

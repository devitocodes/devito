from devito.arch import Cpu64, Intel64, Arm, Power, Device
from devito.core.cpu import (Cpu64NoopCOperator, Cpu64NoopOmpOperator,
                             Cpu64AdvCOperator, Cpu64AdvOmpOperator,
                             Cpu64FsgCOperator, Cpu64FsgOmpOperator,
                             Cpu64CustomOperator, Cpu64CXXCustomOperator,
                             Cpu64CXXNoopCOperator, Cpu64CXXNoopOmpOperator,
                             Cpu64AdvCXXOperator, Cpu64CXXAdvOmpOperator,
                             Cpu64CXXFsgCOperator, Cpu64CXXFsgOmpOperator)

from devito.core.intel import (Intel64AdvCOperator, Intel64AdvOmpOperator,
                               Intel64FsgCOperator, Intel64FsgOmpOperator,
                               Intel64CXXAdvCOperator, Intel64CXXAdvOmpOperator,
                               Intel64CXXFsgCOperator, Intel64CXXFsgOmpOperator)
from devito.core.arm import (ArmAdvCOperator, ArmAdvOmpOperator,
                             ArmCXXAdvCOperator, ArmCXXAdvOmpOperator)
from devito.core.power import (PowerAdvCOperator, PowerAdvOmpOperator,
                               PowerCXXAdvCOperator, PowerCXXAdvOmpOperator)
from devito.core.gpu import (DeviceNoopOmpOperator, DeviceNoopAccOperator,
                             DeviceAdvOmpOperator, DeviceAdvAccOperator,
                             DeviceFsgOmpOperator, DeviceFsgAccOperator,
                             DeviceCustomOmpOperator, DeviceCustomAccOperator)
from devito.operator.registry import operator_registry

# Register CPU Operators
operator_registry.add(Cpu64CustomOperator, Cpu64, 'custom', 'C')
operator_registry.add(Cpu64CustomOperator, Cpu64, 'custom', 'openmp')
operator_registry.add(Cpu64CXXCustomOperator, Cpu64, 'custom', 'CXX')
operator_registry.add(Cpu64CXXCustomOperator, Cpu64, 'custom', 'CXXopenmp')

operator_registry.add(Cpu64NoopCOperator, Cpu64, 'noop', 'C')
operator_registry.add(Cpu64NoopOmpOperator, Cpu64, 'noop', 'openmp')
operator_registry.add(Cpu64CXXNoopCOperator, Cpu64, 'noop', 'CXX')
operator_registry.add(Cpu64CXXNoopOmpOperator, Cpu64, 'noop', 'CXXopenmp')

operator_registry.add(Cpu64AdvCOperator, Cpu64, 'advanced', 'C')
operator_registry.add(Cpu64AdvOmpOperator, Cpu64, 'advanced', 'openmp')
operator_registry.add(Cpu64AdvCXXOperator, Cpu64, 'advanced', 'CXX')
operator_registry.add(Cpu64CXXAdvOmpOperator, Cpu64, 'advanced', 'CXXopenmp')

operator_registry.add(Cpu64FsgCOperator, Cpu64, 'advanced-fsg', 'C')
operator_registry.add(Cpu64FsgOmpOperator, Cpu64, 'advanced-fsg', 'openmp')
operator_registry.add(Cpu64CXXFsgCOperator, Cpu64, 'advanced-fsg', 'CXX')
operator_registry.add(Cpu64CXXFsgOmpOperator, Cpu64, 'advanced-fsg', 'CXXopenmp')

operator_registry.add(Intel64AdvCOperator, Intel64, 'advanced', 'C')
operator_registry.add(Intel64AdvOmpOperator, Intel64, 'advanced', 'openmp')
operator_registry.add(Intel64CXXAdvCOperator, Intel64, 'advanced', 'CXX')
operator_registry.add(Intel64CXXAdvOmpOperator, Intel64, 'advanced', 'CXXopenmp')

operator_registry.add(Intel64FsgCOperator, Intel64, 'advanced-fsg', 'C')
operator_registry.add(Intel64FsgOmpOperator, Intel64, 'advanced-fsg', 'openmp')
operator_registry.add(Intel64CXXFsgCOperator, Intel64, 'advanced-fsg', 'CXX')
operator_registry.add(Intel64CXXFsgOmpOperator, Intel64, 'advanced-fsg', 'CXXopenmp')

operator_registry.add(ArmAdvCOperator, Arm, 'advanced', 'C')
operator_registry.add(ArmAdvOmpOperator, Arm, 'advanced', 'openmp')
operator_registry.add(ArmCXXAdvCOperator, Arm, 'advanced', 'CXX')
operator_registry.add(ArmCXXAdvOmpOperator, Arm, 'advanced', 'CXXopenmp')

operator_registry.add(PowerAdvCOperator, Power, 'advanced', 'C')
operator_registry.add(PowerAdvOmpOperator, Power, 'advanced', 'openmp')
operator_registry.add(PowerCXXAdvCOperator, Power, 'advanced', 'CXX')
operator_registry.add(PowerCXXAdvOmpOperator, Power, 'advanced', 'CXXopenmp')

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

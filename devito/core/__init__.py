"""
The ``core`` Devito backend is simply a "shadow" of the ``base`` backend,
common to all other backends. The ``core`` backend (and therefore the ``base``
backend as well) are used to run Devito on standard CPU architectures.
"""

from devito.archinfo import Cpu64, Intel64, Arm, Power, Device
from devito.parameters import Parameters, add_sub_configuration
from devito.targets import (CPU64Target, Intel64Target, ArmTarget, PowerTarget,
                            CPU64NoopTarget, CustomTarget, DeviceOffloadingTarget,
                            targets)

core_configuration = Parameters('core')
env_vars_mapper = {}
add_sub_configuration(core_configuration, env_vars_mapper)

# Add core-specific Targets
targets.add(CPU64NoopTarget, Cpu64, 'noop')
targets.add(CPU64Target, Cpu64, 'advanced')
targets.add(CustomTarget, Cpu64, 'custom')

targets.add(CPU64NoopTarget, Intel64, 'noop')
targets.add(Intel64Target, Intel64, 'advanced')
targets.add(CustomTarget, Intel64, 'custom')

targets.add(CPU64NoopTarget, Arm, 'noop')
targets.add(ArmTarget, Arm, 'advanced')
targets.add(CustomTarget, Arm, 'custom')

targets.add(CPU64NoopTarget, Power, 'noop')
targets.add(PowerTarget, Power, 'advanced')
targets.add(CustomTarget, Power, 'custom')

targets.add(CPU64NoopTarget, Device, 'noop')
targets.add(DeviceOffloadingTarget, Device, 'advanced')

# The following used by backends.backendSelector
from devito.core.operator import OperatorCore as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.caching import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

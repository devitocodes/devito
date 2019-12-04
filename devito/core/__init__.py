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
targets.add(Cpu64, {'noop': CPU64NoopTarget,
                    'advanced': CPU64Target,
                    'custom': CustomTarget})
targets.add(Intel64, {'noop': CPU64NoopTarget,
                      'advanced': Intel64Target,
                      'custom': CustomTarget})
targets.add(Arm, {'noop': CPU64NoopTarget,
                  'advanced': ArmTarget,
                  'custom': CustomTarget})
targets.add(Power, {'noop': CPU64NoopTarget,
                    'advanced': PowerTarget,
                    'custom': CustomTarget})
targets.add(Device, {'noop': CPU64NoopTarget,
                     'advanced': DeviceOffloadingTarget})

# The following used by backends.backendSelector
from devito.core.operator import OperatorCore as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.caching import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

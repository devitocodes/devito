"""
The ``core`` Devito backend is simply a "shadow" of the ``base`` backend,
common to all other backends. The ``core`` backend (and therefore the ``base``
backend as well) are used to run Devito on standard CPU architectures.
"""

from devito.archinfo import Cpu64, Intel64, Arm, Power, Device
from devito.parameters import Parameters, add_sub_configuration
from devito.targets import (CPU64Rewriter, Intel64Rewriter, ArmRewriter, PowerRewriter,
                            PlatformRewriter, CustomRewriter, DeviceOffloadingRewriter,
                            targets)

core_configuration = Parameters('core')
env_vars_mapper = {}
add_sub_configuration(core_configuration, env_vars_mapper)

# Add core-specific Targets
targets.add(Cpu64, {'noop': PlatformRewriter,
                    'advanced': CPU64Rewriter,
                    'custom': CustomRewriter})
targets.add(Intel64, {'noop': PlatformRewriter,
                      'advanced': Intel64Rewriter,
                      'custom': CustomRewriter})
targets.add(Arm, {'noop': PlatformRewriter,
                  'advanced': ArmRewriter,
                  'custom': CustomRewriter})
targets.add(Power, {'noop': PlatformRewriter,
                    'advanced': PowerRewriter,
                    'custom': CustomRewriter})
targets.add(Device, {'noop': PlatformRewriter,
                     'advanced': DeviceOffloadingRewriter})

# The following used by backends.backendSelector
from devito.core.operator import OperatorCore as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.caching import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

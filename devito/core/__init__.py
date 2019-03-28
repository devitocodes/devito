"""
The ``core`` Devito backend is simply a "shadow" of the ``base`` backend,
common to all other backends. The ``core`` backend (and therefore the ``base``
backend as well) are used to run Devito on standard CPU architectures.
"""

from devito.archinfo import Cpu64, Intel64, Arm, Power
from devito.dle import (CPU64Rewriter, Intel64Rewriter, ArmRewriter, PowerRewriter,
                        SpeculativeRewriter, modes)
from devito.parameters import Parameters, add_sub_configuration

core_configuration = Parameters('core')
env_vars_mapper = {}
add_sub_configuration(core_configuration, env_vars_mapper)

# Add core-specific DLE modes
modes.add(Cpu64, {'advanced': CPU64Rewriter, 'speculative': SpeculativeRewriter})
modes.add(Intel64, {'advanced': Intel64Rewriter, 'speculative': SpeculativeRewriter})
modes.add(Arm, {'advanced': ArmRewriter, 'speculative': SpeculativeRewriter})
modes.add(Power, {'advanced': PowerRewriter, 'speculative': SpeculativeRewriter})

# The following used by backends.backendSelector
from devito.core.operator import OperatorCore as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.basic import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

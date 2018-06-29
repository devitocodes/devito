"""
The ``core`` Devito backend is simply a "shadow" of the ``base`` backend,
common to all other backends. The ``core`` backend (and therefore the ``base``
backend as well) are used to run Devito on standard CPU architectures.
"""

from devito.dle import (BasicRewriter, AdvancedRewriter, AdvancedRewriterSafeMath,
                        SpeculativeRewriter, init_dle)
from devito.parameters import Parameters, add_sub_configuration

core_configuration = Parameters('core')
env_vars_mapper = {}
add_sub_configuration(core_configuration, env_vars_mapper)

# Initialize the DLE
modes = {'basic': BasicRewriter,
         'advanced': AdvancedRewriter,
         'advanced-safemath': AdvancedRewriterSafeMath,
         'speculative': SpeculativeRewriter}
init_dle(modes)

# The following used by backends.backendSelector
from devito.function import *  # noqa
from devito.grid import Grid  # noqa
from devito.core.operator import Operator  # noqa
from devito.types import CacheManager  # noqa

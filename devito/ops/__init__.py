"""
The ``ops`` Devito backend uses the OPS library to generate,
JIT-compile, and run kernels on multiple architectures.
"""

from devito.dle import (BasicRewriter, AdvancedRewriter, AdvancedRewriterSafeMath,
                        SpeculativeRewriter, init_dle)
from devito.parameters import Parameters, add_sub_configuration

ops_configuration = Parameters('ops')
env_vars_mapper = {}
add_sub_configuration(ops_configuration, env_vars_mapper)

# Initialize the DLE
modes = {'basic': BasicRewriter,
         'advanced': AdvancedRewriter,
         'advanced-safemath': AdvancedRewriterSafeMath,
         'speculative': SpeculativeRewriter}
init_dle(modes)

# The following used by backends.backendSelector
from devito.function import *  # noqa
from devito.grid import Grid  # noqa
from devito.ops.operator import Operator  # noqa
from devito.types import CacheManager  # noqa

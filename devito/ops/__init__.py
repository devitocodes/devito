"""
The ``ops`` Devito backend uses the OPS library to generate,
JIT-compile, and run kernels on multiple architectures.
"""

from devito.dle import CPU64Rewriter, SpeculativeRewriter, init_dle
from devito.parameters import Parameters, add_sub_configuration

ops_configuration = Parameters('ops')
env_vars_mapper = {}
add_sub_configuration(ops_configuration, env_vars_mapper)

# Initialize the DLE
modes = {'advanced': CPU64Rewriter,
         'speculative': SpeculativeRewriter}
init_dle(modes)

# The following used by backends.backendSelector
from devito.ops.operator import OperatorOPS as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.basic import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

"""
The ``core`` Devito backend is simply a "shadow" of the ``base`` backend,
common to all other backends. The ``core`` backend (and therefore the ``base``
backend as well) are used to run Devito on standard CPU architectures.
"""

from devito.dle import CPU64Rewriter, SpeculativeRewriter, init_dle
from devito.parameters import Parameters, add_sub_configuration

core_configuration = Parameters('core')
env_vars_mapper = {}
add_sub_configuration(core_configuration, env_vars_mapper)

# Initialize the DLE
modes = {'advanced': CPU64Rewriter,
         'speculative': SpeculativeRewriter}
init_dle(modes)

# The following used by backends.backendSelector
from devito.core.operator import OperatorCore as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.basic import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

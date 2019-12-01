"""
The ``ops`` Devito backend uses the OPS library to generate,
JIT-compile, and run kernels on multiple architectures.
"""

from devito.archinfo import Cpu64
from devito.parameters import Parameters, add_sub_configuration
from devito.targets import CPU64NoopRewriter, targets

# Add OPS-specific Targets
targets.add(Cpu64, {'noop': CPU64NoopRewriter,
                    'advanced': CPU64NoopRewriter})

# The following used by backends.backendSelector
from devito.ops.compiler import CompilerOPS as Compiler # noqa

ops_configuration = Parameters('ops')
ops_configuration.add('compiler', Compiler())
env_vars_mapper = {}
add_sub_configuration(ops_configuration, env_vars_mapper)

from devito.ops.operator import OperatorOPS as Operator  # noqa
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.caching import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

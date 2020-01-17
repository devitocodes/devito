"""
The ``ops`` Devito backend uses the OPS library to generate,
JIT-compile, and run kernels on multiple architectures.
"""

from devito.archinfo import Cpu64
from devito.operator.registry import operator_registry
from devito.parameters import Parameters, add_sub_configuration

from devito.ops.compiler import CompilerOPS as Compiler # noqa

ops_configuration = Parameters('ops')
ops_configuration.add('compiler', Compiler())
env_vars_mapper = {}
add_sub_configuration(ops_configuration, env_vars_mapper)

from devito.ops.operator import OPSOperator  # noqa

# Add OPS-specific Operators
operator_registry.add(OPSOperator, Cpu64, 'noop')
operator_registry.add(OPSOperator, Cpu64, 'advanced')

# The following used by backends.backendSelector
Operator = OPSOperator
from devito.types.constant import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.caching import CacheManager  # noqa
from devito.types.grid import Grid  # noqa

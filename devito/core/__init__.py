"""
The ``core`` Devito backend is simply a "shadow" of the ``base`` backend,
common to all other backends. The ``core`` backend (and therefore the ``base``
backend as well) are used to run Devito on standard CPU architectures.
"""

from devito.parameters import Parameters, add_sub_configuration

core_configuration = Parameters('core')
core_configuration.add('autotuning', 'basic', ['none', 'basic', 'aggressive'])

env_vars_mapper = {
    'DEVITO_AUTOTUNING': 'autotuning',
}

add_sub_configuration(core_configuration, env_vars_mapper)

# The following used by backends.backendSelector
from devito.function import (Constant, Function, TimeFunction, SparseFunction,  # noqa
                             SparseTimeFunction)
from devito.core.operator import Operator  # noqa
from devito.types import CacheManager  # noqa

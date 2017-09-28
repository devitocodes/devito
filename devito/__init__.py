from __future__ import absolute_import
import gc

from sympy.core import cache

from devito.base import *  # noqa
from devito.backends import init_backend
from devito.finite_difference import *  # noqa
from devito.dimension import *  # noqa
from devito.interfaces import Forward, Backward, _SymbolCache  # noqa
from devito.logger import error, warning, info  # noqa
from devito.parameters import (configuration, init_configuration,  # noqa
                               env_vars_mapper)
from devito.tools import *  # noqa
from devito.dse import *  # noqa


def clear_cache():
    cache.clear_cache()
    gc.collect()

    for key, val in list(_SymbolCache.items()):
        if val() is None:
            del _SymbolCache[key]


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions


# Initialize the Devito backend
configuration.add('travis_test', 0, [0, 1], lambda i: bool(i))
configuration.add('autotuning', 'basic', ['none', 'basic', 'aggressive'])
init_configuration()
init_backend(configuration['backend'])


def print_defaults():
    """Print the environment variables accepted by Devito, their default value,
    as well as all of the accepted values."""
    for k, v in env_vars_mapper.items():
        info('%s: %s. Default: %s' % (k, configuration._accepted[v],
                                      configuration._defaults[v]))


def print_state():
    """Print the current configuration state."""
    for k, v in configuration.items():
        info('%s: %s' % (k, v))

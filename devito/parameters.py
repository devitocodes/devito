"""The parameters dictionary contains global parameter settings."""

from collections import OrderedDict
from os import environ

from devito.backends import backends_registry
from devito.compiler import compiler_registry, set_compiler
from devito.dse import modes as dse_registry
from devito.dle import modes as dle_registry
from devito.logger import debug, logger_registry, set_log_level

__all__ = ['configuration', 'init_configuration', 'print_defaults', 'print_state']

# Be EXTREMELY careful when writing to a Parameters dictionary
# Read here for reference: http://wiki.c2.com/?GlobalVariablesAreBad
# https://softwareengineering.stackexchange.com/questions/148108/why-is-global-state-so-evil
# If any issues related to global state arise, the following class should
# be made immutable. It shall only be written to at application startup
# and never modified.


class Parameters(OrderedDict):
    """
    A dictionary-like class to hold global configuration parameters for devito
    On top of a normal dict, this provides the option to provide callback functions
    so that any interested module can be informed when the configuration changes.
    """
    def __init__(self, name=None, **kwargs):
        super(Parameters, self).__init__(**kwargs)
        self._name = name
        self._update_functions = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                self[key] = value

    def __setitem__(self, key, value):
        super(Parameters, self).__setitem__(key, value)
        self._updated(key, value)

    def _updated(self, key, value):
        """
        Call any provided update functions so that the other modules know we've
        been updated.
        """
        if key in self._update_functions:
            self._update_functions[key](value)

    def set_update_function(self, key, callable):
        """
        Make ``callable`` be executed when the value of ``key`` changes.
        """
        assert key in self
        self._update_functions[key] = callable

    def initialize(self):
        """
        Execute all callbacks in ``self._update_functions``. Should be invoked
        once right after all entries have been set.
        """
        for k, v in self.items():
            self._updated(k, v)


configuration = Parameters("Devito-Configuration")
"""The Devito configuration parameters."""

defaults = {
    'backend': 'core',
    'log_level': 'INFO',
    'autotuning': 'basic',
    'compiler': 'custom',
    'openmp': False,
    'dse': 'advanced',
    'dle': 'advanced'
}
"""The default Devito configuration parameters"""

accepted = {
    'backend': list(backends_registry),
    'log_level': list(logger_registry),
    'autotuning': ('none', 'basic', 'aggressive'),
    'compiler': list(compiler_registry),
    'openmp': [1, 0],
    'dse': list(dse_registry),
    'dle': list(dle_registry)
}
"""Accepted values for the Devito environment variables."""

env_vars_mapper = {
    'DEVITO_ARCH': 'compiler',
    'DEVITO_AUTOTUNING': 'autotuning',
    'DEVITO_BACKEND': 'backend',
    'DEVITO_DSE': 'dse',
    'DEVITO_DLE': 'dle',
    'DEVITO_OPENMP': 'openmp',
    'DEVITO_LOGGING': 'log_level'
}


def init_configuration():
    # Populate /parameters/
    if environ.get('DEVITO_CONFIG') is None:
        # Try env variables, otherwise pick defaults
        for k, v in sorted(env_vars_mapper.items()):
            configuration[v] = environ.get(k, defaults[v])
    else:
        # Attempt reading from the specified configuration file
        raise NotImplementedError("Devito doesn't support configuration via file.")

    # Parameters casting and checking
    for k, v in list(configuration.items()):
        try:
            val = int(v)
        except (TypeError, ValueError):
            val = v
        if val not in accepted[k]:
            raise ValueError("Illegal configuration parameter (%s, %s). "
                             "Accepted: %s" % (k, val, str(accepted[k])))
        configuration[k] = val

    # Global setup
    # - Logger
    configuration.set_update_function('log_level', lambda i: set_log_level(i))
    # - Compilation toolchain
    configuration['compiler'] = set_compiler(configuration['compiler'],
                                             configuration['openmp'])
    configuration['openmp'] = bool(configuration['openmp'])

    configuration.initialize()


def print_defaults():
    """Print the environment variables accepted by Devito, their default value,
    as well as all of the accepted values."""
    for k, v in env_vars_mapper.items():
        debug('%s: %s. Default: %s' % (k, accepted[v], defaults[v]))


def print_state():
    """Print the current configuration state."""
    for k, v in configuration.items():
        debug('%s: %s' % (k, v))

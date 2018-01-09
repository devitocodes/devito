"""The parameters dictionary contains global parameter settings."""

from collections import OrderedDict
from os import environ

__all__ = ['configuration', 'init_configuration', 'print_defaults', 'print_state',
           'add_sub_configuration']

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
        self._accepted = {}
        self._defaults = {}
        self._update_functions = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                self[key] = value

    def _check_key_value(func):
        def wrapper(self, key, value):
            accepted = self._accepted[key]
            if accepted is not None:
                tocheck = list(value) if isinstance(value, dict) else [value]
                if any(i not in accepted for i in tocheck):
                    raise ValueError("Illegal configuration parameter (%s, %s). "
                                     "Accepted: %s" % (key, value, str(accepted)))
            func(self, key, value)
        return wrapper

    def _updated(self, key, value):
        """
        Call any provided update functions so that the other modules know we've
        been updated.
        """
        if key in self._update_functions:
            retval = self._update_functions[key](value)
            if retval is not None:
                super(Parameters, self).__setitem__(key, retval)

    @_check_key_value
    def __setitem__(self, key, value):
        super(Parameters, self).__setitem__(key, value)
        self._updated(key, value)

    @_check_key_value
    def update(self, key, value):
        """
        Update the parameter ``key`` to ``value``. This is different from
        ``self[key] = value`` as the callback, if any, is bypassed.
        """
        super(Parameters, self).__setitem__(key, value)

    def add(self, key, value, accepted=None, callback=None):
        """
        Add a new parameter ``key`` with default value ``value``.

        Associate ``key`` with a list of ``accepted`` values.

        If provided, make sure ``callback`` is executed when the value of ``key``
        changes.
        """
        super(Parameters, self).__setitem__(key, value)
        self._accepted[key] = accepted
        self._defaults[key] = value
        if callable(callback):
            self._update_functions[key] = callback

    def initialize(self):
        """
        Execute all callbacks in ``self._update_functions``. Should be invoked
        once right after all entries have been set.
        """
        for k, v in self.items():
            self._updated(k, v)

    @property
    def name(self):
        return self._name


env_vars_mapper = {
    'DEVITO_ARCH': 'compiler',
    'DEVITO_ISA': 'isa',
    'DEVITO_PLATFORM': 'platform',
    'DEVITO_BACKEND': 'backend',
    'DEVITO_DSE': 'dse',
    'DEVITO_DLE': 'dle',
    'DEVITO_DLE_OPTIONS': 'dle_options',
    'DEVITO_OPENMP': 'openmp',
    'DEVITO_LOGGING': 'log_level',
    'DEVITO_FIRST_TOUCH': 'first_touch',
    'DEVITO_DEBUG_COMPILER': 'debug_compiler',
}

configuration = Parameters("Devito-Configuration")
"""The Devito configuration parameters."""


def init_configuration(configuration=configuration, env_vars_mapper=env_vars_mapper):
    # Populate /configuration/ with user-provided options
    if environ.get('DEVITO_CONFIG') is None:
        # Try env variables, otherwise stick to defaults
        unprocessed = OrderedDict([(v, environ.get(k, configuration._defaults[v]))
                                   for k, v in sorted(env_vars_mapper.items())])
    else:
        # Attempt reading from the specified configuration file
        raise NotImplementedError("Devito doesn't support configuration via file yet.")

    # Parameters validation
    for k, v in unprocessed.items():
        try:
            items = v.split(';')
            # Env variable format: 'var=k1:v1;k2:v2:k3:v3:...'
            keys, values = zip(*[i.split(':') for i in items])
            # Casting
            values = [eval(i) for i in values]
        except AttributeError:
            # Env variable format: 'var=v', 'v' is not a string
            keys = [v]
            values = []
        except ValueError:
            # Env variable format: 'var=k1;k2:v2...' or even just 'var=v'
            keys = [i.split(':')[0] for i in items]
            values = []
            # Cast to integer
            for i, j in enumerate(list(keys)):
                try:
                    keys[i] = int(j)
                except (TypeError, ValueError):
                    keys[i] = j
        if len(keys) == len(values):
            configuration.update(k, dict(zip(keys, values)))
        elif len(keys) == 1:
            configuration.update(k, keys[0])
        else:
            configuration.update(k, keys)

    configuration.initialize()


def add_sub_configuration(sub_configuration, sub_env_vars_mapper=None):
    init_configuration(sub_configuration, sub_env_vars_mapper or {})
    # For use from within a backend (i.e., inside Devito)
    setattr(configuration, sub_configuration.name, sub_configuration)
    # For use in user code, when the backend is a runtime choice and some
    # options are in common between the supported backends
    setattr(configuration, 'backend', sub_configuration)


def print_defaults():
    """Print the environment variables accepted by Devito, their default value,
    as well as all of the accepted values."""
    from devito.logger import info
    for k, v in env_vars_mapper.items():
        info('%s: %s. Default: %s' % (k, configuration._accepted[v],
                                      configuration._defaults[v]))


def print_state():
    """Print the current configuration state."""
    from devito.logger import info
    for k, v in configuration.items():
        info('%s: %s' % (k, v))

"""The parameters dictionary contains global parameter settings."""

from collections import OrderedDict
from os import environ
from functools import wraps

from devito.logger import info, warning
from devito.tools import Signer, filter_ordered

__all__ = ['configuration', 'init_configuration', 'print_defaults', 'print_state',
           'switchconfig']

# Be EXTREMELY careful when writing to a Parameters dictionary
# Read here for reference: http://wiki.c2.com/?GlobalVariablesAreBad
# https://softwareengineering.stackexchange.com/questions/148108/why-is-global-state-so-evil
# If any issues related to global state arise, the following class should
# be made immutable. It shall only be written to at application startup
# and never modified.


class Parameters(OrderedDict, Signer):

    """
    A dictionary-like class to hold global configuration parameters for devito
    On top of a normal dict, this provides the option to provide callback functions
    so that any interested module can be informed when the configuration changes.
    """

    def __init__(self, name=None, **kwargs):
        super(Parameters, self).__init__(**kwargs)
        self._name = name

        self._accepted = {}
        self._deprecated = {}
        self._defaults = {}
        self._impact_jit = {}

        self._preprocess_functions = {}
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
            return func(self, key, value)
        return wrapper

    def _check_key_deprecation(func):
        def wrapper(self, key, value=None):
            if key in self._deprecated:
                warning("Trying to access deprecated config `%s`. Using `%s` instead"
                        % (key, self._deprecated[key]))
                key = self._deprecated[key]
            return func(self, key, value)
        return wrapper

    def _preprocess(self, key, value):
        """
        Execute the preprocesser associated to ``key``, if any. This will
        return a new value.
        """
        if key in self._preprocess_functions:
            return self._preprocess_functions[key](value)
        else:
            return value

    def _updated(self, key, value):
        """
        Execute the callback associated to ``key``, if any.
        """
        if key in self._update_functions:
            retval = self._update_functions[key](value)
            if retval is not None:
                super(Parameters, self).__setitem__(key, retval)

    @_check_key_deprecation
    def __getitem__(self, key, *args):
        return super(Parameters, self).__getitem__(key)

    @_check_key_deprecation
    @_check_key_value
    def __setitem__(self, key, value):
        value = self._preprocess(key, value)
        super(Parameters, self).__setitem__(key, value)
        self._updated(key, value)

    @_check_key_deprecation
    @_check_key_value
    def update(self, key, value):
        """
        Update the parameter ``key`` to ``value``. This is different from
        ``self[key] = value`` as both preprocessor and callback, if any,
        are bypassed.
        """
        super(Parameters, self).__setitem__(key, value)

    def add(self, key, value, accepted=None, preprocessor=None, callback=None,
            impacts_jit=True, deprecate=None):
        """
        Add a new parameter ``key`` with default value ``value``.

        Associate ``key`` with a list of ``accepted`` values.

        If provided, make sure ``callback`` is executed when the value of ``key``
        changes.

        If ``impacts_jit`` is False (defaults to True), then it can be assumed
        that the parameter doesn't affect code generation, so it can be excluded
        from the construction of the hash key.
        """
        super(Parameters, self).__setitem__(key, value)
        self._accepted[key] = accepted
        self._defaults[key] = value
        self._impact_jit[key] = impacts_jit
        if callable(preprocessor):
            self._preprocess_functions[key] = preprocessor
        if callable(callback):
            self._update_functions[key] = callback
        if deprecate is not None:
            self._deprecated[deprecate] = key

    def initialize(self):
        """
        Execute all preprocessors and callbacks, thus completing the
        initialization.
        """
        for k, v in list(self.items()):
            # Will trigger preprocessor and callback, if any
            self[k] = v

    @property
    def name(self):
        return self._name

    def _signature_items(self):
        # Note: we are discarding some vars that do not affect the C level
        # code in order to avoid recompiling when such vars are modified
        return tuple(str(sorted((k, v) for k, v in self.items() if self._impact_jit[k])))


env_vars_mapper = {
    'DEVITO_ARCH': 'compiler',
    'DEVITO_PLATFORM': 'platform',
    'DEVITO_PROFILING': 'profiling',
    'DEVITO_DEVELOP': 'develop-mode',
    'DEVITO_OPT': 'opt',
    'DEVITO_MPI': 'mpi',
    'DEVITO_LANGUAGE': 'language',
    'DEVITO_AUTOTUNING': 'autotuning',
    'DEVITO_LOGGING': 'log-level',
    'DEVITO_FIRST_TOUCH': 'first-touch',
    'DEVITO_JIT_BACKDOOR': 'jit-backdoor',
    'DEVITO_IGNORE_UNKNOWN_PARAMS': 'ignore-unknowns',
    'DEVITO_SAFE_MATH': 'safe-math'
}

env_vars_deprecated = {
    'DEVITO_DLE': ('DEVITO_OPT', 'DEVITO_OPT should be used instead'),
    'DEVITO_OPENMP': ('DEVITO_LANGUAGE', 'DEVITO_LANGUAGE=openmp should be used instead'),
}


configuration = Parameters("Devito-Configuration")
"""The Devito configuration parameters."""


def init_configuration(configuration=configuration, env_vars_mapper=env_vars_mapper,
                       env_vars_deprecated=env_vars_deprecated):
    # Populate `configuration` with user-provided options
    if environ.get('DEVITO_CONFIG') is None:
        # It is important to configure `platform`, `compiler`, and the rest, in this order
        process_order = filter_ordered(['platform', 'compiler'] +
                                       list(env_vars_mapper.values()))
        queue = sorted(env_vars_mapper.items(), key=lambda i: process_order.index(i[1]))
        unprocessed = OrderedDict([(v, environ.get(k, configuration._defaults[v]))
                                   for k, v in queue])

        # Handle deprecated env vars
        mapper = dict(queue)
        for k, (v, msg) in env_vars_deprecated.items():
            if environ.get(k):
                warning("`%s` is deprecated. %s" % (k, msg))
                if environ.get(v):
                    warning("Both `%s` and `%s` set. Ignoring `%s`" % (k, v, k))
                else:
                    warning("Setting `%s=%s`" % (v, environ[k]))
                    unprocessed[mapper[v]] = environ[k]
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


class switchconfig(object):

    """
    Decorator to temporarily change `configuration` parameters.
    """

    def __init__(self, **params):
        self.params = {k.replace('_', '-'): v for k, v in params.items()}

    def __call__(self, func, *args, **kwargs):
        @wraps(func)
        def wrapper(*args, **kwargs):
            previous = {}
            for k, v in self.params.items():
                previous[k] = configuration[k]
                configuration[k] = v
            try:
                result = func(*args, **kwargs)
            finally:
                for k, v in self.params.items():
                    try:
                        configuration[k] = previous[k]
                    except ValueError:
                        # E.g., `platform` and `compiler` will end up here
                        configuration[k] = previous[k].name
            return result
        return wrapper


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

from collections import namedtuple
from itertools import product

# Import the global `configuration` dict
from devito.parameters import *  # noqa

# API imports
from devito.base import *  # noqa
from devito.builtins import *  # noqa
from devito.data.allocators import *  # noqa
from devito.equation import *  # noqa
from devito.finite_differences import *  # noqa
from devito.mpi import MPI  # noqa
from devito.types import _SymbolCache, NODE, CELL, Buffer, SubDomain, SubDomainSet  # noqa
from devito.types.dimension import *  # noqa

# Imports required to initialize Devito
from devito.archinfo import platform_registry
from devito.backends import backends_registry, init_backend
from devito.compiler import compiler_registry
from devito.dle import dle_registry
from devito.dse import dse_registry
from devito.logger import error, warning, info, logger_registry, set_log_level  # noqa
from devito.profiling import profiler_registry


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions

# Setup target platform, compiler, and backend
configuration.add('platform', 'cpu64', list(platform_registry),
                  callback=lambda i: platform_registry[i]())
configuration.add('compiler', 'custom', list(compiler_registry),
                  callback=lambda i: compiler_registry[i]())
configuration.add('backend', 'core', list(backends_registry), callback=init_backend)

# Should Devito run a first-touch Operator upon data allocation?
configuration.add('first-touch', 0, [0, 1], lambda i: bool(i), False)

# Should Devito ignore any unknown runtime arguments supplied to Operator.apply(),
# or rather raise an exception (the default behaviour)?
configuration.add('ignore-unknowns', 0, [0, 1], lambda i: bool(i), False)

# Setup log level
configuration.add('log-level', 'INFO', list(logger_registry),
                  lambda i: set_log_level(i), False)

# Escape hatch for custom kernels. The typical use case is as follows: one lets
# Devito generate code for an Operator; then, once the session is over, the
# generated file is manually modified (e.g., for debugging or for performance
# experimentation); finally, when re-running the same program, Devito won't
# overwrite the user-modified files (thus entirely bypassing code generation),
# and will instead use the custom kernel
configuration.add('jit-backdoor', 0, [0, 1], lambda i: bool(i), False)

# Enable/disable automatic padding for allocated data
configuration.add('autopadding', False, [False, True])

# Execution mode setup
def _reinit_compiler(val):  # noqa
    # Force re-build the compiler
    configuration['compiler'].__init__(suffix=configuration['compiler'].suffix,
                                       mpi=configuration['mpi'])
    return bool(val) if isinstance(val, int) else val
configuration.add('openmp', 0, [0, 1], callback=_reinit_compiler)  # noqa
configuration.add('mpi', 0, [0, 1, 'basic', 'diag', 'overlap', 'overlap2', 'full'],
                  callback=_reinit_compiler)

# Autotuning setup
at_levels = ['off', 'basic', 'aggressive', 'max']
at_modes = ['preemptive', 'destructive', 'runtime']
at_default_mode = {'core': 'preemptive', 'yask': 'runtime', 'ops': 'runtime'}
at_setup = namedtuple('at_setup', 'level mode')
at_accepted = at_levels + [list(i) for i in product(at_levels, at_modes)]
def _at_callback(val):  # noqa
    if isinstance(val, str):
        level, mode = val, at_default_mode[configuration['backend']]
    else:
        level, mode = val
    if level == 'off':
        level = False
    return at_setup(level, mode)
configuration.add('autotuning', 'off', at_accepted, callback=_at_callback,  # noqa
                  impacts_jit=False)

# Should Devito emit the JIT compilation commands?
configuration.add('debug-compiler', 0, [0, 1], lambda i: bool(i), False)

# In develop-mode:
# - Some optimizations may not be applied to the generated code.
# - The compiler performs more type and value checking
configuration.add('develop-mode', True, [False, True])

# Setup DSE
configuration.add('dse', 'advanced', list(dse_registry))

# Setup DLE
configuration.add('dle', 'advanced', list(dle_registry))
configuration.add('dle-options', {})

# Setup Operator profiling
configuration.add('profiling', 'basic', list(profiler_registry), impacts_jit=False)

# Initialize `configuration`. This will also trigger the backend initialization
init_configuration()

# Expose a mechanism to clean up the symbol caches (SymPy's, Devito's)
clear_cache = CacheManager().clear  # noqa


# Helper functions to switch on/off optimisation levels
def mode_develop():
    """Run all future Operators in develop mode. This is the default mode."""
    configuration['develop-mode'] = True


def mode_performance():
    """
    Run all future Operators in performance mode. The performance mode
    also employs suitable NUMA strategies for memory allocation.
    """
    configuration['develop-mode'] = False
    configuration['autotuning'] = ['aggressive',
                                   at_default_mode[configuration['backend']]]
    # With the autotuner in `aggressive` mode, a more aggressive blocking strategy
    # which also tiles the innermost loop) is beneficial
    configuration['dle-options']['blockinner'] = True

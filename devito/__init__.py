from collections import namedtuple
from itertools import product

from devito.archinfo import known_isas, known_platforms, get_isa, get_platform
from devito.base import *  # noqa
from devito.builtins import *  # noqa
from devito.data.allocators import *  # noqa
from devito.equation import *  # noqa
from devito.finite_differences import *  # noqa
from devito.logger import error, warning, info, set_log_level  # noqa
from devito.parameters import *  # noqa
from devito.types import NODE, CELL, Buffer, SubDomain  # noqa
from devito.types.dimension import *  # noqa

from devito.compiler import compiler_registry
from devito.backends import backends_registry, init_backend


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions

# Setup compiler and backend
configuration.add('compiler', 'custom', list(compiler_registry),
                  callback=lambda i: compiler_registry[i]())
configuration.add('backend', 'core', list(backends_registry), callback=init_backend)

# Should Devito run a first-touch Operator upon data allocation?
configuration.add('first-touch', 0, [0, 1], lambda i: bool(i), False)

# Should Devito ignore any unknown runtime arguments supplied to Operator.apply(),
# or rather raise an exception (the default behaviour)?
configuration.add('ignore-unknowns', 0, [0, 1], lambda i: bool(i), False)

# Escape hatch for custom kernels. The typical use case is as follows: one lets
# Devito generate code for an Operator; then, once the session is over, the
# generated file is manually modified (e.g., for debugging or for performance
# experimentation); finally, when re-running the same program, Devito won't
# overwrite the user-modified files (thus entirely bypassing code generation),
# and will instead use the custom kernel
configuration.add('jit-backdoor', 0, [0, 1], lambda i: bool(i), False)

# (Undocumented) escape hatch for cross-compilation
configuration.add('cross-compile', None)

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
AT_LEVELs = ['off', 'basic', 'aggressive']
AT_MODEs = ['preemptive', 'destructive', 'runtime']
at_default_mode = {'core': 'preemptive', 'yask': 'runtime', 'ops': 'runtime'}
at_setup = namedtuple('at_setup', 'level mode')
at_accepted = AT_LEVELs + [list(i) for i in product(AT_LEVELs, AT_MODEs)]
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

# Instruction Set Architecture (ISA)
configuration.add('isa', get_isa(), known_isas)

# Codename of the underlying architecture
configuration.add('platform', get_platform(), known_platforms)

# In develop-mode:
# - Some optimizations may not be applied to the generated code.
# - The compiler performs more type and value checking
configuration.add('develop-mode', True, [False, True])

# Initialize the configuration, either from the environment or
# defaults. This will also trigger the backend initialization
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


def mode_benchmark():
    """Like ``mode_performance``, but also switch YASK's autotuner mode to
    ``preemptive``."""
    mode_performance()
    configuration['autotuning'] = ['aggressive', 'preemptive']

import os
from subprocess import PIPE, Popen
from collections import namedtuple
from itertools import product

import cpuinfo

from devito.base import *  # noqa
from devito.builtins import *  # noqa
from devito.data.allocators import *  # noqa
from devito.equation import *  # noqa
from devito.finite_differences import *  # noqa
from devito.logger import error, warning, info, set_log_level  # noqa
from devito.parameters import *  # noqa
from devito.tools import *  # noqa
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

# By default, the Devito compiler generates parameters, rather than numbers, for
# things such as array casts, loop bounds, etc. This maximises Operator reusability,
# as the same Operator can be applied to Functions that only different in the shape.
# It is also the only viable way when using MPI. One can change this behaviour
# (e.g., for educational purposes) by playing with the `codegen` configuration knob
configuration.add('codegen', 'parametric', ['parametric', 'explicit'])

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
configuration.add('mpi', 0, [0, 1, 'basic', 'diag', 'diag_wmsg'],
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

# Set the Instruction Set Architecture (ISA)
ISAs = ['cpp', 'avx', 'avx2', 'avx512']
configuration.add('isa', 'cpp', ISAs)

# Set the CPU architecture (only codename)
PLATFORMs = ['intel64', 'snb', 'ivb', 'hsw', 'bdw', 'skx', 'knl']
configuration.add('platform', 'intel64', PLATFORMs)


def infer_cpu():
    """
    Detect the highest Instruction Set Architecture and the platform
    codename using cpu flags and/or leveraging other tools. Return default
    values if the detection procedure was unsuccesful.
    """
    cpu_info = cpuinfo.get_cpu_info()
    # ISA
    isa = configuration._defaults['isa']
    for i in reversed(configuration._accepted['isa']):
        if any(j.startswith(i) for j in cpu_info['flags']):
            # Using `startswith`, rather than `==`, as a flag such as 'avx512'
            # appears as 'avx512f, avx512cd, ...'
            isa = i
            break
    # Platform
    try:
        # First, try leveraging `gcc`
        p1 = Popen(['gcc', '-march=native', '-Q', '--help=target'], stdout=PIPE)
        p2 = Popen(['grep', 'march'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
        output, _ = p2.communicate()
        platform = output.decode("utf-8").split()[1]
        # Full list of possible /platform/ values at this point at:
        # https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
        platform = {'sandybridge': 'snb', 'ivybridge': 'ivb', 'haswell': 'hsw',
                    'broadwell': 'bdw', 'skylake': 'skx', 'knl': 'knl'}[platform]
    except:
        # Then, try infer from the brand name, otherwise fallback to default
        try:
            platform = cpu_info['brand'].split()[4]
            platform = {'v2': 'ivb', 'v3': 'hsw', 'v4': 'bdw', 'v5': 'skx'}[platform]
        except:
            platform = None
    # Is it a known platform?
    if platform not in configuration._accepted['platform']:
        platform = configuration._defaults['platform']
    return isa, platform


# In develop-mode:
# - Some optimizations may not be applied to the generated code.
# - The compiler performs more type and value checking
def _switch_cpu(develop_mode):
    if bool(develop_mode) is False:
        isa, platform = infer_cpu()
        configuration['isa'] = os.environ.get('DEVITO_ISA', isa)
        configuration['platform'] = os.environ.get('DEVITO_PLATFORM', platform)
    else:
        configuration['isa'] = 'cpp'
        configuration['platform'] = 'intel64'
configuration.add('develop-mode', True, [False, True], _switch_cpu)  # noqa

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


def mode_benchmark():
    """Like ``mode_performance``, but also switch YASK's autotuner mode to
    ``preemptive``."""
    mode_performance()
    configuration['autotuning'] = ['aggressive', 'preemptive']

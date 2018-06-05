from __future__ import absolute_import

import os
from subprocess import PIPE, Popen

import cpuinfo

from devito.base import *  # noqa
from devito.data import *  # noqa
from devito.dimension import *  # noqa
from devito.equation import *  # noqa
from devito.finite_difference import *  # noqa
from devito.logger import error, warning, info, set_log_level, silencio  # noqa
from devito.parameters import *  # noqa
from devito.tools import *  # noqa

from devito.compiler import compiler_registry
from devito.backends import backends_registry, init_backend


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions

# First add the compiler configuration option...
configuration.add('compiler', 'custom', list(compiler_registry),
                  callback=lambda i: compiler_registry[i]())


def _cast_and_update_compiler(val):
    # Force re-build the compiler
    configuration['compiler'].__init__(suffix=configuration['compiler'].suffix)
    return bool(val)


configuration.add('openmp', 0, [0, 1], callback=_cast_and_update_compiler)
configuration.add('debug_compiler', 0, [0, 1], lambda i: bool(i))

# ... then the backend configuration. The order is important since the
# backend might depend on the compiler configuration.
configuration.add('backend', 'core', list(backends_registry),
                  callback=init_backend)

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

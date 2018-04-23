from __future__ import absolute_import

import os

from devito.base import *  # noqa
from devito.data import *  # noqa
from devito.dimension import *  # noqa
from devito.equation import *  # noqa
from devito.finite_difference import *  # noqa
from devito.grid import *  # noqa
from devito.logger import error, warning, info, set_log_level  # noqa
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

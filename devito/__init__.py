from __future__ import absolute_import
import gc

from sympy.core import cache

from devito.base import *  # noqa
from devito.finite_difference import *  # noqa
from devito.dimension import *  # noqa
from devito.grid import *  # noqa
from devito.function import Forward, Backward  # noqa
from devito.types import _SymbolCache  # noqa
from devito.logger import error, warning, info  # noqa
from devito.parameters import *  # noqa
from devito.tools import *  # noqa
from devito.dse import *  # noqa

from devito.compiler import compiler_registry
from devito.backends import backends_registry, init_backend


def clear_cache():
    cache.clear_cache()
    gc.collect()

    for key, val in list(_SymbolCache.items()):
        if val() is None:
            del _SymbolCache[key]


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions

# First add the compiler configuration option...
configuration.add('compiler', 'custom', list(compiler_registry),
                  lambda i: compiler_registry[i]())
configuration.add('openmp', 0, [0, 1], lambda i: bool(i))
configuration.add('debug_compiler', 0, [0, 1], lambda i: bool(i))

# ... then the backend configuration. The order is important since the
# backend might depend on the compiler configuration.
configuration.add('backend', 'core', list(backends_registry),
                  callback=init_backend)

# Set the Instruction Set Architecture (ISA)
ISAs = [None, 'cpp', 'avx', 'avx2', 'avx512', 'knc']
configuration.add('isa', None, ISAs)

# Set the CPU architecture (only codename)
PLATFORMs = [None, 'intel64', 'sandybridge', 'ivybridge', 'haswell',
             'broadwell', 'skylake', 'knc', 'knl']
# TODO: switch arch to actual architecture names; use the mapper in /YASK/
configuration.add('platform', None, PLATFORMs)


# Initialize the configuration, either from the environment or
# defaults. This will also trigger the backend initialization
init_configuration()

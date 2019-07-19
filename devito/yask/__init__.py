"""
The ``yask`` Devito backend uses the YASK stencil optimizer to generate,
JIT-compile, and run kernels.
"""

import os
import sys

from devito.archinfo import Arm, Cpu64, CPU64, Power
from devito.dle import PlatformRewriter, modes
from devito.exceptions import InvalidOperator
from devito.logger import yask as log
from devito.parameters import Parameters, configuration, add_sub_configuration
from devito.tools import make_tempdir

from devito.yask.dle import YaskRewriter
from devito.yask.utils import namespace


def exit(emsg):
    """
    Handle fatal errors.
    """
    raise InvalidOperator("YASK Error [%s]. Exiting..." % emsg)


log("Backend initialization...")

# Not all devito `platform`s are supported by YASK
if isinstance(configuration['platform'], (Arm, Power)):
    raise ValueError("The YASK backend doesn't support platform `%s`" %
                     configuration['platform'])
# Some of the supported devito `platform`s (e.g., AMDs) may still be run through
# YASK -- as they are x86-64 just like Intels -- but a proper `Platform` must be used
if configuration['platform'] is CPU64:
    configuration['platform'] = 'intel64'

try:
    import yask as yc
    # YASK compiler factories
    cfac = yc.yc_factory()
    nfac = yc.yc_node_factory()
    ofac = yc.yask_output_factory()
except ImportError:
    exit("Python YASK compiler bindings")

path = os.path.dirname(os.path.dirname(yc.__file__))
namespace['path'] = path
namespace['kernel-path'] = os.path.join(path, 'src', 'kernel')
namespace['yask-output-dir'] = make_tempdir('yask')
# The YASK compiler expects the generated code in
# $YASK_OUTPUT_DIR/build/kernel/$stencil.$arch/gen/yask_stencil_code.hpp
# unless overridden by the YK_CODE_FILE make var
namespace['yask-lib'] = os.path.join(namespace['yask-output-dir'], 'lib')
namespace['yask-pylib'] = os.path.join(namespace['yask-output-dir'], 'yask')
namespace['yask-codegen'] = lambda i, j, k: os.path.join(namespace['yask-output-dir'],
                                                         'build', 'kernel',
                                                         '%s.%s.%s' % (i, j, k), 'gen')
namespace['yask-codegen-file'] = 'yask_stencil_code.hpp'

# All dynamically generated Python modules are stored here
os.makedirs(namespace['yask-pylib'], exist_ok=True)
with open(os.path.join(namespace['yask-pylib'], '__init__.py'), 'w') as f:
    f.write('')
sys.path.append(os.path.join(namespace['yask-pylib']))


# Need a custom compiler to compile YASK kernels
# This is derived from the user-selected compiler
class YaskCompiler(configuration['compiler'].__class__):

    def __init__(self, *args, **kwargs):
        kwargs['cpp'] = True
        kwargs['suffix'] = configuration['compiler'].suffix
        super(YaskCompiler, self).__init__(*args, **kwargs)
        self.cflags = [i for i in configuration['compiler'].cflags
                       if not i.startswith('-std')] + ['-std=c++11']
        # Tell the compiler where to get YASK header files and shared objects
        self.include_dirs.append(os.path.join(namespace['path'], 'include'))
        self.library_dirs.append(namespace['yask-lib'])
        self.ldflags.append('-Wl,-rpath,%s' % namespace['yask-lib'])


yask_configuration = Parameters('yask')
yask_configuration.add('compiler', YaskCompiler())
callback = lambda i: eval(i) if i else ()
yask_configuration.add('folding', (), callback=callback, impacts_jit=False)
yask_configuration.add('blockshape', (), callback=callback, impacts_jit=False)
yask_configuration.add('clustering', (), callback=callback, impacts_jit=False)
yask_configuration.add('options', None, impacts_jit=False)
yask_configuration.add('dump', None, impacts_jit=False)

env_vars_mapper = {
    'DEVITO_YASK_FOLDING': 'folding',
    'DEVITO_YASK_BLOCKING': 'blockshape',
    'DEVITO_YASK_CLUSTERING': 'clustering',
    'DEVITO_YASK_OPTIONS': 'options',
    'DEVITO_YASK_DUMP': 'dump'
}

add_sub_configuration(yask_configuration, env_vars_mapper)

# Add YASK-specific DLE modes
modes.add(Cpu64, {'noop': PlatformRewriter,
                  'advanced': YaskRewriter,
                  'speculative': YaskRewriter})

# The following used by backends.backendSelector
from devito.types import SparseFunction, SparseTimeFunction  # noqa
from devito.yask.types import CacheManager, Grid, Constant, Function, TimeFunction  # noqa
from devito.yask.operator import OperatorYASK as Operator  # noqa

log("Backend successfully initialized!")

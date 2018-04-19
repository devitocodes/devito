"""
The ``yask`` Devito backend uses the YASK stencil optimizer to generate,
JIT-compile, and run kernels.
"""

from collections import OrderedDict
import os

from devito.dle import BasicRewriter, init_dle
from devito.exceptions import InvalidOperator
from devito.logger import yask as log
from devito.parameters import Parameters, configuration, add_sub_configuration
from devito.tools import ctypes_pointer, infer_cpu

from devito.yask.dle import YaskRewriter


def exit(emsg):
    """
    Handle fatal errors.
    """
    raise InvalidOperator("YASK Error [%s]. Exiting..." % emsg)


log("Backend initialization...")
try:
    import yask as yc
    # YASK compiler factories
    cfac = yc.yc_factory()
    nfac = yc.yc_node_factory()
    ofac = yc.yask_output_factory()
except ImportError:
    exit("Python YASK compiler bindings")

path = os.path.dirname(os.path.dirname(yc.__file__))

# YASK conventions
namespace = OrderedDict()
namespace['jit-yc-hook'] = lambda i, j: 'devito_%s_yc_hook%d' % (i, j)
namespace['jit-yk-hook'] = lambda i, j: 'devito_%s_yk_hook%d' % (i, j)
namespace['jit-yc-soln'] = lambda i, j: 'devito_%s_yc_soln%d' % (i, j)
namespace['jit-yk-soln'] = lambda i, j: 'devito_%s_yk_soln%d' % (i, j)
namespace['kernel-filename'] = 'yask_stencil_code.hpp'
namespace['path'] = path
namespace['kernel-path'] = os.path.join(path, 'src', 'kernel')
namespace['kernel-path-gen'] = os.path.join(namespace['kernel-path'], 'gen')
namespace['kernel-output'] = os.path.join(namespace['kernel-path-gen'],
                                          namespace['kernel-filename'])
namespace['code-soln-type'] = 'yask::yk_solution'
namespace['code-soln-name'] = 'soln'
namespace['code-soln-run'] = 'run_solution'
namespace['code-grid-type'] = 'yask::yk_grid'
namespace['code-grid-name'] = lambda i: "grid_%s" % str(i)
namespace['code-grid-get'] = 'get_element'
namespace['code-grid-put'] = 'set_element'
namespace['type-solution'] = ctypes_pointer('yask::yk_solution_ptr')
namespace['type-grid'] = ctypes_pointer('yask::yk_grid_ptr')


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
        self.library_dirs.append(os.path.join(namespace['path'], 'lib'))
        self.ldflags.append('-Wl,-rpath,%s' % os.path.join(namespace['path'], 'lib'))


yask_configuration = Parameters('yask')
yask_configuration.add('compiler', YaskCompiler())
callback = lambda i: eval(i) if i else ()
yask_configuration.add('autotuning', 'runtime', ['off', 'runtime', 'preemptive'])
yask_configuration.add('folding', (), callback=callback)
yask_configuration.add('blockshape', (), callback=callback)
yask_configuration.add('clustering', (), callback=callback)
yask_configuration.add('options', None)
yask_configuration.add('dump', None)


# In develop-mode, no optimizations are applied to the generated code (e.g., SIMD).
# When switching to non-develop-mode, optimizations are automatically switched on,
# sniffing the highest Instruction Set Architecture level available on the architecture
def switch_cpu(develop_mode):
    if bool(develop_mode) is False:
        isa, platform = infer_cpu()
        configuration['isa'] = os.environ.get('DEVITO_ISA', isa)
        configuration['platform'] = os.environ.get('DEVITO_PLATFORM', platform)
    else:
        configuration['isa'] = 'cpp'
        configuration['platform'] = 'intel64'
yask_configuration.add('develop-mode', True, [False, True], switch_cpu)  # noqa

env_vars_mapper = {
    'DEVITO_YASK_DEVELOP': 'develop-mode',
    'DEVITO_YASK_AUTOTUNING': 'autotuning',
    'DEVITO_YASK_FOLDING': 'folding',
    'DEVITO_YASK_BLOCKING': 'blockshape',
    'DEVITO_YASK_CLUSTERING': 'clustering',
    'DEVITO_YASK_OPTIONS': 'options',
    'DEVITO_YASK_DUMP': 'dump'
}

add_sub_configuration(yask_configuration, env_vars_mapper)

# Initialize the DLE
modes = {'basic': BasicRewriter,
         'advanced': YaskRewriter,
         'advanced-safemath': YaskRewriter,
         'speculative': YaskRewriter}
init_dle(modes)

# The following used by backends.backendSelector
from devito.yask.function import Constant, Function, TimeFunction  # noqa
from devito.function import SparseFunction, SparseTimeFunction  # noqa
from devito.yask.operator import Operator  # noqa
from devito.yask.types import CacheManager  # noqa

log("Backend successfully initialized!")

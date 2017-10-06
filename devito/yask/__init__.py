"""
The ``yask`` Devito backend uses the YASK stencil optimizer to generate,
JIT-compile, and run kernels.
"""

from collections import OrderedDict
import os

import cpuinfo

from devito import configuration
from devito.exceptions import InvalidOperator
from devito.logger import yask as log
from devito.parameters import Parameters
from devito.tools import ctypes_pointer


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
try:
    # Set directories for generated code
    path = os.environ['YASK_HOME']
except KeyError:
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
namespace['time-dim'] = 't'
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
        super(YaskCompiler, self).__init__(*args, **kwargs)
        # Switch to C++
        self.cc = self.cpp_mapper[configuration['compiler'].cc]
        self.ld = self.cpp_mapper[configuration['compiler'].ld]
        self.cflags = configuration['compiler'].cflags + ['-std=c++11']
        self.src_ext = 'cpp'
        # Tell the compiler where to get YASK header files and shared objects
        self.include_dirs.append(os.path.join(namespace['path'], 'include'))
        self.library_dirs.append(os.path.join(namespace['path'], 'lib'))
        self.ldflags.append('-Wl,-rpath,%s' % os.path.join(namespace['path'], 'lib'))


yask_configuration = Parameters('YASK-Configuration')
yask_configuration.add('compiler', YaskCompiler())
yask_configuration.add('python-exec', False, [False, True])
# Set the Instruction Set Architecture used by the YASK code generator
default_isa = 'cpp'
ISAs = ['cpp', 'avx', 'avx2', 'avx512', 'knc']
yask_configuration.add('isa', default_isa, ISAs)
# Currently YASK also require the CPU architecture (e.g., snb for sandy bridge,
# hsw for haswell, etc.). At the moment, we simply infer it from the ISA
arch_mapper = {'cpp': 'intel64', 'avx': 'snb', 'avx2': 'hsw', 'avx512': 'knl'}
yask_configuration.add('arch', arch_mapper[default_isa], arch_mapper.values())


# In develop-mode, no optimizations are applied to the generated code (e.g., SIMD)
# When switching to non-develop-mode, optimizations are automatically switched on,
# sniffing the highest Instruction Set Architecture level available on the current
# machine and providing it to YASK
def reset_yask_isa(develop_mode):
    isa = default_isa
    if develop_mode is False:
        cpu_flags = cpuinfo.get_cpu_info()['flags']
        for i in reversed(ISAs):
            if i in cpu_flags:
                isa = i
                break
    yask_configuration['isa'] = isa
    yask_configuration['arch'] = arch_mapper[isa]
yask_configuration.add('develop-mode', True, [False, True], reset_yask_isa)  # noqa

configuration.add('yask', yask_configuration)

log("Backend successfully initialized!")


# The following used by backends.backendSelector
from devito.yask.interfaces import ConstantData, DenseData, TimeData  # noqa
from devito.pointdata import PointData  # noqa
from devito.yask.operator import Operator  # noqa

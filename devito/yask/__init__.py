"""
The ``yask`` Devito backend uses the YASK stencil optimizer to generate,
JIT-compile, and run kernels.
"""

from collections import OrderedDict
import os

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
    import yask_compiler as yc
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
    exit("Missing YASK_HOME")

# YASK conventions
namespace = OrderedDict()
namespace['kernel-hook'] = 'hook'
namespace['kernel-real'] = 'kernel'
namespace['kernel-filename'] = 'yask_stencil_code.hpp'
namespace['path'] = path
namespace['kernel-path'] = os.path.join(path, 'src', 'kernel')
namespace['kernel-path-gen'] = os.path.join(namespace['kernel-path'], 'gen')
namespace['kernel-output'] = os.path.join(namespace['kernel-path-gen'],
                                          namespace['kernel-filename'])
namespace['time-dim'] = 't'
namespace['type-solution'] = ctypes_pointer('yask::yk_solution_ptr')
namespace['type-grid'] = ctypes_pointer('yask::yk_grid_ptr')
namespace['code-grid'] = lambda i: "grid_%s" % str(i)
namespace['code-soln'] = 'soln'


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
# TODO: this should be somewhat sniffed
yask_configuration.add('arch', 'snb', ['snb'])
yask_configuration.add('isa', 'cpp', ['cpp'])
configuration.add('yask', yask_configuration)

log("Backend successfully initialized!")


# The following used by backends.backendSelector
from devito.yask.interfaces import ConstantData, DenseData, TimeData  # noqa
from devito.pointdata import PointData  # noqa
from devito.yask.operator import Operator  # noqa

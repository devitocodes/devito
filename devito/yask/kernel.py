from collections import OrderedDict
import os
import importlib

from devito.compiler import make
from devito.exceptions import CompilationError, InvalidOperator
from devito.logger import info


class YaskContext(object):

    def __init__(self, arch, isa, cfac, nfac, ofac):
        """
        Proxy between Devito and YASK.

        A new YaskContext is required wheneven any of the following change: ::

            * YASK version
            * Target architecture (``arch``) and instruction set (``isa``)
            * Floating-point precision (``dtype``)
            * Domain dimensions (``dim_sizes``)
            * Folding
            * Grid memory layout scheme

        :param arch: Architecture identifier, in a format understood by YASK.
        :param isa: Instruction set architecture, in a format understood by YASK.
        :param cfac: YASK compiler factory.
        :param nfac: YASK node factory, to create ASTs.
        :param ofac: YASK compiler output factory, to handle output streams.
        """
        self.arch = arch
        self.isa = isa
        self.cfac = cfac
        self.nfac = nfac
        self.ofac = ofac

        self._kernels = OrderedDict()

        self.env = None
        self.dim_sizes = None
        self.dtype = None
        self.hook_soln = None

        self._initialized = False

    def __finalize__(self, env, dim_sizes, dtype, hook_soln):
        """
        Finalize instantiation.

        :param env: Global environment (e.g., MPI).
        :param dim_sizes: Domain size along each dimension.
        :param dtype: The data type used in kernels, as a NumPy dtype.
        :param hook_soln: "Fake" solution to track YASK grids.
        """
        self.env = env
        self.dim_sizes = dim_sizes
        self.dtype = dtype
        self.hook_soln = hook_soln

        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    @property
    def space_dimensions(self):
        return self.hook_soln.get_domain_dim_names()

    @property
    def time_dimension(self):
        return self.hook_soln.get_step_dim_name()

    @property
    def dimensions(self):
        return (self.time_dimension,) + self.space_dimensions

    @property
    def space_shape(self):
        ret = []
        for k, v in self.dim_sizes.items():
            if k in self.space_dimensions:
                ret.append(v)
        return tuple(ret)

    @property
    def shape(self):
        return tuple(self.dim_sizes.values())

    @property
    def grids(self):
        mapper = {}
        if not self.initialized:
            return mapper
        for i in range(self.hook_soln.get_num_grids()):
            grid = self.hook_soln.get_grid(i)
            mapper[grid.get_name()] = grid
        return mapper

    @property
    def nkernels(self):
        return len(self._kernels)

    @property
    def ngrids(self):
        return len(self.grids)

    def add_grid(self, name, dimensions):
        """
        Add and return a new grid ``name``. If a grid ``name`` already exists,
        then simply return it, without further actions.
        """
        grids = self.grids
        if name in grids:
            return grids[name]
        else:
            # new_grid() also modifies the /hook_soln/ state
            grid = self.hook_soln.new_grid(name, *dimensions)
            # Allocate memory
            self.hook_soln.prepare_solution()
            return grid

    def add_kernel(self, key, kernel):
        """
        Add a new ``kernel`` uniquely identified by ``key``.
        """
        self._kernels[key] = kernel

    def get_kernel(self, key):
        """
        Retrieve the kernel idenfified by ``key``.
        """
        return self._kernels[key]


def init(dimensions, shape, dtype):
    """
    Create a new :class:`YaskContext`.

    To be called prior to any YASK-related operation.
    """

    if YASK.initialized:
        return

    info("Initializing YASK [kernel API]")

    assert len(dimensions) == len(shape)

    # Create a new stencil solution
    soln = cfac.new_solution(namespace['kernel-hook'])

    # Silence YASK
    soln.set_debug_output(ofac.new_string_output())

    # Setup hook solution builder
    soln.set_step_dim_name(namespace['time-dim'])
    dimensions = [str(i) for i in dimensions]
    if set(dimensions) < {'x', 'y', 'z'}:
        _force_exit("Need a DenseData[x,y,z] for initialization")
    # TODO: YASK only accepts x,y,z
    soln.set_domain_dim_names(*[i for i in dimensions if i != namespace['time-dim']])

    # Number of bytes in each FP value
    soln.set_element_bytes(dtype().itemsize)

    # JIT YASK kernel
    yk = yask_jit(soln, namespace['kernel-hook'])

    # YASK Hook kernel factory
    kfac = yk.yk_factory()

    # Initalize MPI, etc
    env = kfac.new_env()

    # Create hook solution
    hook_soln = kfac.new_solution(env)

    # Silence YASK
    hook_soln.set_debug_output(yk.yask_output_factory().new_string_output())

    # Setup hook solution
    dim_sizes = OrderedDict(zip(dimensions, shape))
    for dm in hook_soln.get_domain_dim_names():
        ds = dim_sizes[dm]
        # Set domain size in each dim.
        hook_soln.set_rank_domain_size(dm, ds)
        # TODO: Add something like: hook_soln.set_min_pad_size(dm, 16)
        # Set block size to 64 in z dim and 32 in other dims.

    # Simple rank configuration in 1st dim only.
    # In production runs, the ranks would be distributed along all domain dimensions.
    # TODO Improve me
    hook_soln.set_num_ranks(hook_soln.get_domain_dim_name(0), env.get_num_ranks())

    # Finish off YASK initialization
    YASK.__finalize__(env, dim_sizes, dtype, hook_soln)

    info("YASK backend successfully initialized!")


def yask_jit(soln, base):
    """
    Write out YASK kernel and create a shared object using YASK's Makefile.

    Import and return the SWIG-created Python module using the shared object.
    """
    assert isinstance(soln, yc.yc_solution)

    # It's necessary to `clean` the YASK kernel directory *before*
    # writing out the first `yask_stencil_code.hpp`
    make(path, ['-C', namespace['kernel-path'], 'clean'])

    # Write out the stencil file
    if not os.path.exists(namespace['kernel-path-gen']):
        os.makedirs(namespace['kernel-path-gen'])
    soln.format(isa, YASK.ofac.new_file_output(namespace['kernel-output']))

    # JIT-compile it
    try:
        make(os.environ['YASK_HOME'], ['-j', 'YK_CXXOPT=-O0',
                                       "EXTRA_MACROS=TRACE",
                                       'YK_BASE=%s' % str(base),
                                       'stencil=%s' % soln.get_name(),
                                       '-C', namespace['kernel-path'], 'api'])
    except CompilationError:
        _force_exit("Hook solution compilation")

    # Import the corresponding Python (SWIG-generated) module
    try:
        return importlib.import_module(base)
    except ImportError:
        _force_exit("Python YASK kernel bindings")


def _force_exit(emsg):
    """
    Handle fatal errors.
    """
    raise InvalidOperator("YASK Error [%s]. Exiting..." % emsg)


# YASK initialization (will be finished by the first call to init())

info("Initializing YASK [compiler API]")

try:
    import yask_compiler as yc
    # YASK compiler factories
    cfac = yc.yc_factory()
    nfac = yc.yc_node_factory()
    ofac = yc.yask_output_factory()
except ImportError:
    _force_exit("Python YASK compiler bindings")
try:
    # Set directories for generated code
    path = os.environ['YASK_HOME']
except KeyError:
    _force_exit("Missing YASK_HOME")

# YASK conventions
namespace = OrderedDict()
namespace['kernel-hook'] = 'devito_hook'
namespace['kernel-real'] = 'devito_kernel'
namespace['kernel-filename'] = 'yask_stencil_code.hpp'
namespace['path'] = path
namespace['kernel-path'] = os.path.join(path, 'src', 'kernel')
namespace['kernel-path-gen'] = os.path.join(namespace['kernel-path'], 'gen')
namespace['kernel-output'] = os.path.join(namespace['kernel-path-gen'],
                                          namespace['kernel-filename'])
namespace['time-dim'] = 't'

# TODO: this should be moved into /configuration/
arch = 'hsw'
isa = 'cpp'

# TODO: only a single YaskContext is assumed at the moment
YASK = YaskContext(arch, isa, cfac, nfac, ofac)

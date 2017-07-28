from collections import OrderedDict
import os
import importlib

from cached_property import cached_property

from devito.compiler import make
from devito.exceptions import CompilationError, InvalidOperator
from devito.logger import yask as log


class YaskContext(object):

    def __init__(self, env, core_sizes, pad_sizes, dtype, hook_soln):
        """
        Proxy between Devito and YASK.

        A new YaskContext is required wheneven any of the following change: ::

            * YASK version
            * Target architecture (``arch``) and instruction set (``isa``)
            * Floating-point precision (``dtype``)
            * Domain dimensions (``domain_sizes``) and padding dimensions (``pad_sizes``)
            * Folding
            * Grid memory layout scheme

        :param env: Global environment (e.g., MPI).
        :param core_sizes: Domain size along each dimension; includes time dimension
        :param pad_sizes: Padding size along each dimension.
        :param dtype: The data type used in kernels, as a NumPy dtype.
        :param hook_soln: "Fake" solution to track YASK grids.
        """
        self.env = env
        self.core_sizes = core_sizes
        self.pad_sizes = pad_sizes
        self.dtype = dtype
        self.hook_soln = hook_soln

        self._kernels = OrderedDict()

    @cached_property
    def space_dimensions(self):
        return self.hook_soln.get_domain_dim_names()

    @cached_property
    def time_dimension(self):
        return self.hook_soln.get_step_dim_name()

    @cached_property
    def dimensions(self):
        return (self.time_dimension,) + self.space_dimensions

    @cached_property
    def domain_sizes(self):
        ret = OrderedDict()
        for k, v in self.core_sizes.items():
            if k in self.space_dimensions:
                ret[k] = v
        return ret

    @cached_property
    def dim_shape(self):
        ret = OrderedDict()
        for (k1, v1), (k2, v2) in zip(self.core_sizes.items(), self.pad_sizes.items()):
            assert k1 == k2
            ret[k1] = v1 + v2*2
        return ret

    @cached_property
    def space_shape(self):
        return tuple(v for k, v in self.dim_shape.items() if k in self.space_dimensions)

    @cached_property
    def shape(self):
        return tuple(self.dim_shape.values())

    @property
    def grids(self):
        mapper = {}
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
            for i in self.space_dimensions:
                grid.set_halo_size(i, self.pad_sizes[i])
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


contexts = OrderedDict()
"""All known YASK contexts."""


def yask_context(dimensions, shape, dtype, space_order):
    """
    Create a new :class:`YaskContext`, or retrieve an existing one with same
    ``dimensions``, ``shape``, and ``dtype``.
    """

    key = (dimensions, shape, dtype, space_order)
    if key in contexts:
        return contexts[key]

    log("Creating new context...")

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

    # Create hook solution
    kfac = yk.yk_factory()
    env = kfac.new_env()
    hook_soln = kfac.new_solution(env)
    hook_soln.set_debug_output(yk.yask_output_factory().new_string_output())

    # Calculate inner and padding regions
    # TODO: This will probably require using NBPML
    core_sizes = OrderedDict()
    pad_sizes = OrderedDict()
    for i, j in zip(dimensions, shape):
        if namespace['time-dim'] != i:
            # Padding only meaningful in space dimensions
            pad_sizes[i] = space_order
            core_sizes[i] = j - pad_sizes[i]*2
        else:
            pad_sizes[i] = 0
            core_sizes[i] = j

    # Setup domain and padding regions in the hook solution
    for i in hook_soln.get_domain_dim_names():
        hook_soln.set_rank_domain_size(i, core_sizes[i])

    # Simple rank configuration in 1st dim only.
    # In production runs, the ranks would be distributed along all domain dimensions.
    # TODO Improve me
    hook_soln.set_num_ranks(hook_soln.get_domain_dim_name(0), env.get_num_ranks())

    contexts[key] = YaskContext(env, core_sizes, pad_sizes, dtype, hook_soln)

    log("Context successfully created!")

    return contexts[key]


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
    soln.format(isa, ofac.new_file_output(namespace['kernel-output']))

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

log("Backend initialization...")

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

log("Backend successfully initialized!")

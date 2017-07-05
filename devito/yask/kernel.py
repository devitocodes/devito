import os

from devito.compiler import make
from devito.exceptions import CompilationError, DLEException
from devito.logger import dle


class YaskKernel(object):

    def __init__(self, cfac, nfac, path):
        """
        A proxy between Devito and YASK.

        :param cfac: YASK compiler factory, to create Solutions.
        :param nfac: YASK node factory, to create ASTs.
        :param path: Generated code dump directory.
        :param env: Global environment (e.g., MPI).
        :param shape: Domain size along each dimension.
        :param dtype: The data type used in kernels, as a NumPy dtype.
        :param hook_soln: "Fake" solution to track YASK grids.
        """
        self.cfac = cfac
        self.nfac = nfac
        self.path = path

        self.env = None
        self.shape = None
        self.dtype = None
        self.hook_soln = None

        self._initialized = False

    def __finalize__(self, env, shape, dtype, hook_soln):
        self.env = env
        self.shape = shape
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
    def grids(self):
        mapper = {}
        for i in range(self.hook_soln.get_num_grids()):
            grid = self.hook_soln.get_grid(i)
            mapper[grid.get_name()] = grid
        return mapper

    def setdefault(self, name, dimensions):
        """
        Add and return a new grid ``name``. If a grid ``name`` already exists,
        then return it without performing any other actions.
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


def init(dimensions, shape, dtype, architecture='hsw', isa='avx2'):
    """
    To be called prior to any YASK-related operation.

    A new bootstrap is required wheneven any of the following change: ::

        * YASK version
        * Target architecture (``architecture`` param)
        * Floating-point precision (``dtype`` param)
        * Domain dimensions (``dimensions`` param)
        * Folding
        * Grid memory layout scheme
    """

    if YASK.initialized:
        return

    dle("Initializing YASK [kernel API]")

    # Create a new stencil solution
    soln = cfac.new_solution("Hook")
    soln.set_step_dim_name("t")

    dimensions = [str(i) for i in dimensions]
    if any(i not in ['x', 'y', 'z'] for i in dimensions):
        _force_exit("Need a DenseData[x,y,z] for initialization")
    soln.set_domain_dim_names(*dimensions)  # TODO: YASK only accepts x,y,z

    # Number of bytes in each FP value
    soln.set_element_bytes(dtype().itemsize)

    # Generate YASK output
    soln.write(os.path.join(path, 'yask_stencil_code.hpp'), isa, True)

    # Build YASK output, and load the corresponding YASK kernel
    try:
        make(os.environ['YASK_HOME'],
             ['-j', 'stencil=Hook', 'arch=%s' % architecture, 'yk-api'])
    except CompilationError:
        _force_exit("Hook solution compilation")
    try:
        import yask_kernel as yk
    except ImportError:
        _force_exit("Python YASK kernel bindings")

    # YASK Hook kernel factory
    kfac = yk.yk_factory()

    # Initalize MPI, etc
    env = kfac.new_env()

    # Create hook solution
    hook_soln = kfac.new_solution(env)
    for dm, ds in zip(hook_soln.get_domain_dim_names(), shape):
        # Set domain size in each dim.
        hook_soln.set_rank_domain_size(dm, ds)
        # TODO: Add something like: hook_soln.set_min_pad_size(dm, 16)
        # Set block size to 64 in z dim and 32 in other dims.
        hook_soln.set_block_size(dm, min(64 if dm == "z" else 32, ds))

    # Simple rank configuration in 1st dim only.
    # In production runs, the ranks would be distributed along all domain dimensions.
    # TODO Improve me
    hook_soln.set_num_ranks(hook_soln.get_domain_dim_name(0), env.get_num_ranks())

    # Finish off YASK initialization
    YASK.__finalize__(env, shape, dtype, hook_soln)

    dle("YASK backend successfully initialized!")


def _force_exit(emsg):
    """
    Handle fatal errors.
    """
    raise DLEException("YASK Error [%s]. Exiting..." % emsg)


# YASK initialization (will be finished by the first call to init())

dle("Initializing YASK [compiler API]")

try:
    import yask_compiler as yc
    # YASK compiler factories
    cfac = yc.yc_factory()
    nfac = yc.yc_node_factory()
except ImportError:
    _force_exit("Python YASK compiler bindings")
try:
    # Set directory for generated code
    path = os.path.join(os.environ['YASK_HOME'], 'src', 'kernel', 'gen')
    if not os.path.exists(path):
        os.makedirs(path)
except KeyError:
    _force_exit("Missing YASK_HOME")
YASK = YaskKernel(cfac, nfac, path)

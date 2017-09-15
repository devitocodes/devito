import os
import sys
import importlib
from collections import OrderedDict

from cached_property import cached_property

import ctypes
import numpy as np

from devito.compiler import make
from devito.exceptions import CompilationError
from devito.logger import yask as log
from devito.tools import numpy_to_ctypes

from devito.yask import cfac, ofac, namespace, exit, yask_configuration
from devito.yask.utils import convert_multislice


class YaskGrid(object):

    """
    A ``YaskGrid`` wraps a YASK grid.

    An implementation of an array that behaves similarly to a ``numpy.ndarray``,
    suitable for the YASK storage layout.

    Subclassing ``numpy.ndarray`` would have led to shadow data copies, because
    of the different storage layout.
    """

    # Force __rOP__ methods (OP={add,mul,...) to get arrays, not scalars, for efficiency
    __array_priority__ = 1000

    def __init__(self, grid, dimensions, shape, halo, dtype):
        """
        Initialize a new :class:`YaskGrid`.
        """
        self.grid = grid
        self.dimensions = dimensions
        self.shape = shape
        self.halo = halo
        self.dtype = dtype

        # Initialize the grid content to 0.
        self._reset()

    def __getitem__(self, index):
        # TODO: ATM, no MPI support.
        start, stop, shape = convert_multislice(index, self.shape, self.halo)
        if not shape:
            log("YaskGrid: Getting single entry %s" % str(start))
            assert start == stop
            out = self.grid.get_element(*start)
        else:
            log("YaskGrid: Getting full-array/block via index [%s]" % str(index))
            out = np.empty(shape, self.dtype, 'C')
            self.grid.get_elements_in_slice(out.data, start, stop)
        return out

    def __setitem__(self, index, val):
        # TODO: ATM, no MPI support.
        start, stop, shape = convert_multislice(index, self.shape, self.halo, 'set')
        if all(i == 1 for i in shape):
            log("YaskGrid: Setting single entry %s" % str(start))
            assert start == stop
            self.grid.set_element(val, *start)
        elif isinstance(val, np.ndarray):
            log("YaskGrid: Setting full-array/block via index [%s]" % str(index))
            self.grid.set_elements_in_slice(val, start, stop)
        elif all(i == j-1 for i, j in zip(shape, self.shape)):
            log("YaskGrid: Setting full-array to given scalar via single grid sweep")
            self.grid.set_all_elements_same(val)
        else:
            log("YaskGrid: Setting block to given scalar via index [%s]" % str(index))
            self.grid.set_elements_in_slice_same(val, start, stop, True)

    def __getslice__(self, start, stop):
        if stop == sys.maxint:
            # Emulate default NumPy behaviour
            stop = None
        return self.__getitem__(slice(start, stop))

    def __setslice__(self, start, stop, val):
        if stop == sys.maxint:
            # Emulate default NumPy behaviour
            stop = None
        self.__setitem__(slice(start, stop), val)

    def __getattr__(self, name):
        """Proxy to yk::grid methods."""
        return getattr(self.grid, name)

    def __repr__(self):
        return repr(self[:])

    def __meta_binop(op):
        # Used to build all binary operations such as __eq__, __add__, etc.
        # These all boil down to calling the numpy equivalents
        def f(self, other):
            return getattr(self[:], op)(other)
        return f
    __eq__ = __meta_binop('__eq__')
    __ne__ = __meta_binop('__ne__')
    __le__ = __meta_binop('__le__')
    __lt__ = __meta_binop('__lt__')
    __ge__ = __meta_binop('__ge__')
    __gt__ = __meta_binop('__gt__')
    __add__ = __meta_binop('__add__')
    __radd__ = __meta_binop('__add__')
    __sub__ = __meta_binop('__sub__')
    __rsub__ = __meta_binop('__sub__')
    __mul__ = __meta_binop('__mul__')
    __rmul__ = __meta_binop('__mul__')
    __div__ = __meta_binop('__div__')
    __rdiv__ = __meta_binop('__div__')
    __truediv__ = __meta_binop('__truediv__')
    __rtruediv__ = __meta_binop('__truediv__')
    __mod__ = __meta_binop('__mod__')
    __rmod__ = __meta_binop('__mod__')

    def _reset(self):
        """
        Reset grid value to 0.
        """
        self[:] = 0.0

    @property
    def name(self):
        return self.grid.get_name()

    @property
    def dimensions(self):
        return self.grid.get_dim_names()

    @property
    def ndpointer(self):
        """Return a :class:`numpy.ndarray` view of the grid content."""
        ctype = numpy_to_ctypes(self.dtype)
        cpointer = ctypes.cast(int(self.grid.get_raw_storage_buffer()),
                               ctypes.POINTER(ctype))
        ndpointer = np.ctypeslib.ndpointer(dtype=self.dtype, shape=self.shape)
        casted = ctypes.cast(cpointer, ndpointer)
        ndarray = np.ctypeslib.as_array(casted, shape=self.shape)
        return ndarray

    @property
    def rawpointer(self):
        return ctypes.cast(int(self.grid), ctypes.c_void_p)

    def give_storage(self, target):
        """
        Share self's storage with ``target``.
        """
        for i in self.dimensions:
            if i == namespace['time-dim']:
                target.set_alloc_size(i, self.get_alloc_size(i))
            else:
                target.set_halo_size(i, self.get_halo_size(i))
        target.share_storage(self.grid)

    def view(self):
        """
        View of the YASK grid in standard (i.e., Devito) row-major layout.
        """
        return self[:]


class YaskSolution(object):

    """
    A ``YaskSolution`` wraps a YASK solution.
    """

    def __init__(self, name, ycsoln):
        """
        Write out a YASK kernel, build it using YASK's Makefiles,
        import the corresponding SWIG-generated Python module, and finally
        create a YASK kernel solution object.

        :param name: Unique name of this YaskSolution.
        :param ycsoln: YaskCompiler solution
        """
        # It's necessary to `clean` the YASK kernel directory *before*
        # writing out the first `yask_stencil_code.hpp`
        make(namespace['path'], ['-C', namespace['kernel-path'], 'clean'])

        # Write out the stencil file
        if not os.path.exists(namespace['kernel-path-gen']):
            os.makedirs(namespace['kernel-path-gen'])
        ycsoln.format(yask_configuration['isa'],
                      ofac.new_file_output(namespace['kernel-output']))

        # JIT-compile it
        try:
            make(os.environ['YASK_HOME'], ['-j', 'YK_CXXOPT=-O0',
                                           "EXTRA_MACROS=TRACE",
                                           'YK_BASE=%s' % str(name),
                                           'stencil=%s' % ycsoln.get_name(),
                                           '-C', namespace['kernel-path'], 'api'])
        except CompilationError:
            exit("Kernel solution compilation")

        # Import the corresponding Python (SWIG-generated) module
        try:
            yk = importlib.import_module(name)
        except ImportError:
            exit("Python YASK kernel bindings")

        # Create the YASK solution object
        kfac = yk.yk_factory()
        self.env = kfac.new_env()
        self.soln = kfac.new_solution(self.env)

        # MPI setup
        self.set_num_ranks()

        # Redirect stdout/strerr to a string
        self.output = yk.yask_output_factory().new_string_output()
        self.soln.set_debug_output(self.output)

        self.name = name

        # Shared object name
        self.soname = "%s.%s.%s" % (name, ycsoln.get_name(), yask_configuration['arch'])

    def set_num_ranks(self):
        """
        Simple rank configuration in 1st dim only.

        This is work-in-progress: in production runs, the ranks would be
        distributed along all domain dimensions.
        """
        self.soln.set_num_ranks(self.space_dimensions[0], self.env.get_num_ranks())

    def set_rank_domain_size(self, domain_sizes):
        for i in self.soln.get_domain_dim_names():
            self.soln.set_rank_domain_size(i, domain_sizes[i])

    def new_grid(self, name, dimensions):
        return self.soln.new_grid(name, *dimensions)

    def prepare(self):
        self.soln.prepare_solution()

    def run(self, ntimesteps):
        self.soln.run_solution(ntimesteps)

    @property
    def space_dimensions(self):
        return tuple(self.soln.get_domain_dim_names())

    @property
    def time_dimension(self):
        return self.soln.get_step_dim_name()

    @property
    def grids(self):
        return self.soln.get_grids()

    @property
    def rawpointer(self):
        return ctypes.cast(int(self.soln), ctypes.c_void_p)

    def __repr__(self):
        return "YaskSolution [%s]" % self.name


class YaskContext(object):

    def __init__(self, name, dimensions, core, halo, dtype, hook):
        """
        Proxy between Devito and YASK.

        A YaskContext is required for any single kernel executed through YASK.

        :param name: Unique name of the context.
        :param dimensions: Context dimensions (may include time dimension).
        :param core: Domain size along each dimension; includes time dimension.
        :param halo: Halo size along each dimension.
        :param dtype: The data type used in kernels, as a NumPy dtype.
        :param hook: "Fake" solution to track YASK grids.
        """
        self.name = name

        self.dimensions = tuple(dimensions)
        self.core = tuple(core)
        self.halo = tuple(halo)

        self.dtype = dtype
        self.hook = hook

        # All known solutions and grids in this context
        self.solutions = []
        self.grids = {}

    @cached_property
    def space_dimensions(self):
        return self.hook.space_dimensions

    @cached_property
    def time_dimension(self):
        return self.hook.time_dimension

    @cached_property
    def dim_core(self):
        return OrderedDict([(i, j) for i, j in zip(self.dimensions, self.core)])

    @cached_property
    def dim_halo(self):
        return OrderedDict([(i, j) for i, j in zip(self.dimensions, self.halo)])

    @cached_property
    def dim_shape(self):
        return OrderedDict([(d, i + j*2) for d, i, j in
                            zip(self.dimensions, self.core, self.halo)])

    @cached_property
    def domain_sizes(self):
        ret = OrderedDict()
        for k, v in self.dim_core.items():
            if k in self.space_dimensions:
                ret[k] = v
        return ret

    @cached_property
    def shape(self):
        return tuple(self.dim_shape.values())

    @property
    def nsolutions(self):
        return len(self.solutions)

    @property
    def ngrids(self):
        return len(self.grids)

    def make_grid(self, name, dimensions, shape, space_order, dtype):
        """
        Create and return a new :class:`YaskGrid`, which wraps a YASK grid.
        """
        # Set up the YASK grid
        grid = self.hook.new_grid(name, dimensions)
        for i in self.space_dimensions:
            grid.set_halo_size(i, self.dim_halo[i])
        if grid.is_dim_used(self.time_dimension):
            grid.set_alloc_size(self.time_dimension, shape[0])

        # Allocate memory immediately as the user may simply want to use it
        if name in self.grids:
            log("Reusing pre-existing grid %s (reinitialized to 0.)" % name)
            self.grids[name]._reset()
        else:
            log("Allocating YaskGrid for %s (%s)" % (name, str(shape)))
            grid.alloc_storage()
            self.grids[name] = YaskGrid(grid, dimensions, shape, self.halo, dtype)

        return self.grids[name]

    def make_solution(self, ycsoln):
        """
        Create and return a new :class:`YaskSolution` using ``self`` as context
        and ``ycsoln`` as YASK compiler ("stencil") solution.
        """
        soln = YaskSolution('%s_soln%d' % (self.name, self.nsolutions), ycsoln)

        # Setup soln's domains
        soln.set_rank_domain_size(self.domain_sizes)

        # Setup soln's grids using the hook solution
        for sgrid in soln.grids:
            name = sgrid.get_name()
            try:
                hgrid = self.grids[name]
            except KeyError:
                exit("Unknown grid %s" % name)
            # Halo in the space dimensions
            for i in self.space_dimensions:
                sgrid.set_halo_size(i, hgrid.get_halo_size(i))
            # Extent of the time dimension
            if sgrid.is_dim_used(self.time_dimension):
                sgrid.set_alloc_size(self.time_dimension,
                                     hgrid.get_alloc_size(self.time_dimension))

        self.solutions.append(soln)

        return soln

    def __repr__(self):
        return ("YaskContext [%s]\n"
                "- core: %s\n"
                "- halo: %s\n"
                "- grids: %s\n"
                "- solns: %s\n") % (self.name, str(self.dim_core), str(self.dim_halo),
                                    ', '.join([str(i) for i in list(self.grids)]),
                                    ', '.join([i.name for i in list(self.solutions)]))


class YaskNullSolution(object):

    """Used when an Operator doesn't actually have a YASK-offloadable tree."""

    def __init__(self):
        self.name = 'null solution'

    def prepare(self):
        pass

    def run(self, ntimesteps):
        exit("Cannot run a NullSolution through YASK's Python bindings")

    @property
    def grids(self):
        return ()


class YaskNullContext(object):

    """Used when an Operator doesn't actually have a YASK-offloadable tree."""

    @property
    def space_dimensions(self):
        return '?'

    @property
    def time_dimension(self):
        return '?'


contexts = OrderedDict()
"""All known YASK contexts."""


def yask_context(dimensions, shape, dtype, space_order):
    """
    Create a new :class:`YaskContext`, or retrieve an existing one with same
    ``dimensions``, ``shape``, ``dtype``, and ``space_order``.
    """

    key = (dimensions, shape, dtype, space_order)
    if key in contexts:
        return contexts[key]

    log("Creating new context...")

    assert len(dimensions) == len(shape)

    # Create a new stencil solution
    soln = cfac.new_solution(namespace['kernel-hook'])

    # Silence YASK
    soln.set_debug_output(ofac.new_null_output())

    # Setup hook solution builder
    soln.set_step_dim_name(namespace['time-dim'])
    dimensions = [str(i) for i in dimensions]
    if set(dimensions) < {'x', 'y', 'z'}:
        exit("Need a DenseData[x,y,z] for initialization")
    # TODO: YASK only accepts x,y,z
    soln.set_domain_dim_names(*[i for i in dimensions if i != namespace['time-dim']])

    # Number of bytes in each FP value
    soln.set_element_bytes(dtype().itemsize)

    # Create hook solution, JIT-ting the corresponding YASK kernel
    hook = YaskSolution(namespace['kernel-hook'], soln)

    # Setup hook solution
    # TODO: This will probably require using NBPML
    core = []
    halo = []
    for i, j in zip(dimensions, shape):
        if namespace['time-dim'] != i:
            # Padding only meaningful in space dimensions
            halo.append(space_order)
            core.append(j - space_order*2)
        else:
            halo.append(0)
            core.append(j)
    hook.set_rank_domain_size(dict(zip(dimensions, core)))

    contexts[key] = YaskContext('devito_ctx%d' % len(contexts),
                                dimensions, core, halo, dtype, hook)

    log("Context successfully created!")

    return contexts[key]

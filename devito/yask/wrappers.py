import os
import sys
import importlib
from glob import glob
from subprocess import call
from collections import OrderedDict

from cached_property import cached_property

import ctypes
import numpy as np

from devito.compiler import make
from devito.exceptions import CompilationError
from devito.logger import yask as log
from devito.tools import numpy_to_ctypes

from devito.yask import cfac, nfac, ofac, namespace, exit, yask_configuration
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

    def __init__(self, grid, shape, radius, dtype):
        """
        Initialize a new :class:`YaskGrid`.

        The storage layout adopted by YASK is as follows: ::

            --------------------------------------------------------------
            | extra_padding | halo |              | halo | extra_padding |
            ------------------------    domain    ------------------------
            |       padding        |              |       padding        |
            --------------------------------------------------------------
            |                         allocation                         |
            --------------------------------------------------------------

        :param grid: The YASK yk::grid that will be wrapped. Data storage will be
                     allocated if not yet available.
        :param shape: The "visibility region" of the YaskGrid. The shape should be
                      at least as big as the domain (in each dimension). If larger,
                      then users will be allowed to access more data entries,
                      such as those lying on the halo region.
        :param radius: The extent of the halo region.
        :param dtype: The type of the raw data.
        """
        self.grid = grid
        self.shape = shape
        self.dtype = dtype

        if not self.is_storage_allocated():
            # Allocate memory in YASK-land and initialize it to 0
            for i, j in zip(self.dimensions, shape):
                if i == namespace['time-dim']:
                    assert self.grid.is_dim_used(i)
                    self.grid.set_alloc_size(i, j)
                else:
                    # Note, from the YASK docs:
                    # "If the halo is set to a value larger than the padding size,
                    # the padding size will be automatically increase to accomodate it."
                    self.grid.set_halo_size(i, radius)
            self.grid.alloc_storage()
            self._reset()

    def __getitem__(self, index):
        start, stop, shape = convert_multislice(index, self.shape, self._offsets)
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
        start, stop, shape = convert_multislice(index, self.shape, self._offsets, 'set')
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
    def _halo(self):
        return [0 if i == namespace['time-dim'] else self.get_halo_size(i)
                for i in self.dimensions]

    @property
    def _padding(self):
        return [0 if i == namespace['time-dim'] else self.get_pad_size(i)
                for i in self.dimensions]

    @property
    def _offsets(self):
        offsets = []
        for i, j in zip(self.dimensions, self._padding):
            ofs = 0 if i == namespace['time-dim'] else self.get_first_rank_alloc_index(i)
            offsets.append(ofs + j)
        return offsets

    @property
    def with_halo(self):
        """
        Return a new wrapper to self's YASK grid in which the halo has been
        unmasked. This allows the caller to write/read the halo region as well as
        the domain.
        """
        return YaskGridWithHalo(self.grid, self.shape, 0, self.dtype)

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


class YaskGridWithHalo(YaskGrid):

    """A helper class for YaskGrid wrappers providing access to the halo region."""

    def __init__(self, grid, shape, radius, dtype):
        super(YaskGridWithHalo, self).__init__(grid, shape, radius, dtype)
        self.shape = [i + 2*j for i, j in zip(self.shape, self._halo)]

    @property
    def _offsets(self):
        offsets = super(YaskGridWithHalo, self)._offsets
        return [i - j for i, j in zip(offsets, self._halo)]


class YaskKernel(object):

    """
    A ``YaskKernel`` wraps a YASK kernel solution.
    """

    def __init__(self, name, yc_soln, domain):
        """
        Write out a YASK kernel, build it using YASK's Makefiles,
        import the corresponding SWIG-generated Python module, and finally
        create a YASK kernel solution object.

        :param name: Unique name of this YaskKernel.
        :param yc_soln: YaskCompiler solution.
        :param domain: A mapper from space dimensions to their domain size.
        """
        self.name = name

        # Shared object name
        self.soname = "%s.%s.%s" % (name, yc_soln.get_name(), yask_configuration['arch'])

        # It's necessary to `clean` the YASK kernel directory *before*
        # writing out the first `yask_stencil_code.hpp`
        make(namespace['path'], ['-C', namespace['kernel-path'], 'clean'])

        # Write out the stencil file
        if not os.path.exists(namespace['kernel-path-gen']):
            os.makedirs(namespace['kernel-path-gen'])
        yc_soln.format(yask_configuration['isa'],
                       ofac.new_file_output(namespace['kernel-output']))

        # JIT-compile it
        try:
            opt_level = 1 if yask_configuration['develop-mode'] else 3
            make(os.environ['YASK_HOME'], ['-j', 'YK_CXXOPT=-O%d' % opt_level,
                                           # "EXTRA_MACROS=TRACE",
                                           'YK_BASE=%s' % str(name),
                                           'stencil=%s' % yc_soln.get_name(),
                                           'arch=%s' % yask_configuration['arch'],
                                           '-C', namespace['kernel-path'], 'api'])
        except CompilationError:
            exit("Kernel solution compilation")

        # Import the corresponding Python (SWIG-generated) module
        try:
            yk = importlib.import_module(name)
        except ImportError:
            exit("Python YASK kernel bindings")
        try:
            yk = reload(yk)
        except NameError:
            # Python 3.5 compatibility
            yk = importlib.reload(yk)

        # Create the YASK solution object
        kfac = yk.yk_factory()
        self.env = kfac.new_env()
        self.soln = kfac.new_solution(self.env)

        # MPI setup: simple rank configuration in 1st dim only.
        # TODO: in production runs, the ranks would be distributed along all
        # domain dimensions.
        self.soln.set_num_ranks(self.soln.get_domain_dim_names()[0],
                                self.env.get_num_ranks())

        # Redirect stdout/strerr to a string
        self.output = yk.yask_output_factory().new_string_output()
        self.soln.set_debug_output(self.output)

        # Set up the solution domain size
        for k, v in domain.items():
            self.soln.set_rank_domain_size(k, v)

    def new_grid(self, obj_name, grid_name, dimensions):
        """Create a new YASK grid."""
        return self.soln.new_grid(grid_name, *dimensions)

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
        return {i.get_name(): i for i in self.soln.get_grids()}

    @property
    def rawpointer(self):
        return ctypes.cast(int(self.soln), ctypes.c_void_p)

    def __repr__(self):
        return "YaskKernel [%s]" % self.name


class YaskContext(object):

    def __init__(self, name, domain, dtype):
        """
        Proxy between Devito and YASK.

        A YaskContext contains N YaskKernel and M YaskGrids.
        Solutions and grids have in common the context domain. Grids, however, may
        differ in the halo region, due to a different space order. The same grid
        could be used in more than one of the N solutions.

        :param name: Unique name of the context.
        :param domain: A mapper from space dimensions to their domain size.
        :param dtype: The data type used in kernels, as a NumPy dtype.
        """
        self.name = name
        self.domain = domain
        self.dtype = dtype

        # All known solutions and grids in this context
        self.solutions = []
        self.grids = {}

        # Build the hook kernel solution (wrapper) to create grids
        yc_hook = self.make_yc_solution(namespace['jit-yc-hook'])
        # Also add a dummy grid to make YASK happy
        dimensions = [nfac.new_step_index(namespace['time-dim'])]
        dimensions += [nfac.new_domain_index(i) for i in domain]
        yc_hook.new_grid('dummy', dimensions)
        self.yk_hook = YaskKernel(namespace['jit-yk-hook'](name, 0), yc_hook, domain)

    @cached_property
    def space_dimensions(self):
        return tuple(self.yk_hook.space_dimensions)

    @cached_property
    def time_dimension(self):
        return self.yk_hook.time_dimension

    @cached_property
    def dimensions(self):
        return (self.time_dimension,) + self.space_dimensions

    @property
    def nsolutions(self):
        return len(self.solutions)

    @property
    def ngrids(self):
        return len(self.grids)

    def make_grid(self, obj):
        """
        Create and return a new :class:`YaskGrid`, a YASK grid wrapper. Memory
        is allocated.

        :param obj: The symbolic data object for which a YASK grid is allocated.
        """
        dimensions = [str(i) for i in obj.indices]
        if set(dimensions) < set(self.space_dimensions):
            exit("Need a DenseData[x,y,z] to create a YASK grid.")
        name = 'devito_%s_%d' % (obj.name, contexts.ngrids)
        log("Allocating YaskGrid for %s (%s)" % (obj.name, str(obj.shape)))
        grid = self.yk_hook.new_grid(obj.name, name, dimensions)
        wrapper = YaskGrid(grid, obj.shape, obj.space_order, obj.dtype)
        self.grids[name] = wrapper
        return wrapper

    def make_yc_solution(self, namer):
        """
        Create and return a YASK compiler solution object.
        """
        yc_soln = cfac.new_solution(namer(self.name, self.nsolutions))
        yc_soln.set_debug_output(ofac.new_null_output())
        yc_soln.set_element_bytes(self.dtype().itemsize)
        return yc_soln

    def make_yk_solution(self, namer, yc_soln):
        """
        Create and return a new :class:`YaskKernel` using ``self`` as context
        and ``yc_soln`` as YASK compiler ("stencil") solution.
        """
        soln = YaskKernel(namer(self.name, self.nsolutions), yc_soln, self.domain)
        self.solutions.append(soln)
        return soln

    def __repr__(self):
        return ("YaskContext: %s\n"
                "- domain: %s\n"
                "- grids: [%s]\n"
                "- solns: [%s]\n") % (self.name, str(self.domain),
                                      ', '.join([i for i in list(self.grids)]),
                                      ', '.join([i.name for i in self.solutions]))


class ContextManager(OrderedDict):

    def __init__(self, *args, **kwargs):
        super(ContextManager, self).__init__(*args, **kwargs)
        self.ncontexts = 0

    def dump(self):
        """
        Drop all known contexts and clean up lib directory.
        """
        self.clear()
        call(['rm', '-f'] + glob(os.path.join(namespace['path'], 'lib', '*devito*')))
        call(['rm', '-f'] + glob(os.path.join(namespace['path'], 'lib', '*hook*')))

    def fetch(self, dimensions, shape, dtype):
        """
        Fetch the :class:`YaskContext` in ``self`` uniquely identified by
        ``dimensions``, ``shape``, and ``dtype``. Create a new (empty)
        :class:`YaskContext` on miss.
        """
        # Sanity checks
        assert len(dimensions) == len(shape)
        dimensions = [str(i) for i in dimensions]
        if set(dimensions) < {'x', 'y', 'z'}:
            exit("Need a DenseData[x,y,z] for initialization")

        # The time dimension is dropped as implicit to the context
        domain = OrderedDict([(i, j) for i, j in zip(dimensions, shape)
                              if i != namespace['time-dim']])

        # A unique key for this context.
        key = tuple([yask_configuration['isa'], dtype] + domain.items())

        # Fetch or create a YaskContext
        if key in self:
            log("Fetched existing context from cache")
        else:
            self[key] = YaskContext('ctx%d' % self.ncontexts, domain, dtype)
            self.ncontexts += 1
            log("Context successfully created!")
        return self[key]

    @property
    def ngrids(self):
        return sum(i.ngrids for i in self.values())


contexts = ContextManager()
"""All known YASK contexts."""


# Helpers

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
        return {}


class YaskNullContext(object):

    """Used when an Operator doesn't actually have a YASK-offloadable tree."""

    @property
    def space_dimensions(self):
        return '?'

    @property
    def time_dimension(self):
        return '?'

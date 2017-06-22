import os
import sys

import numpy as np
from sympy import Indexed

from devito.compiler import make
from devito.dimension import LoweredDimension
from devito.dle import retrieve_iteration_tree
from devito.dle.backends import BasicRewriter, dle_pass
from devito.exceptions import CompilationError, DLEException
from devito.logger import debug, dle, dle_warning
from devito.visitors import FindSymbols
from devito.tools import as_tuple

__all__ = ['YaskRewriter', 'init', 'YaskGrid']


YASK = None
"""Global state for generation of YASK kernels."""


class YaskState(object):

    def __init__(self, cfac, nfac, path, env, shape, dtype, hook_soln):
        """
        Global state to interact with YASK.

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
        self.env = env
        self.shape = shape
        self.dtype = dtype
        self.hook_soln = hook_soln

    @property
    def dimensions(self):
        return self.hook_soln.get_domain_dim_names()

    @property
    def grids(self):
        mapper = {}
        for i in range(self.hook_soln.get_num_grids()):
            grid = self.hook_soln.get_grid(i)
            mapper[grid.get_name()] = grid
        return mapper

    def setdefault(self, name):
        """
        Add and return a new grid ``name``. If a grid ``name`` already exists,
        then return it without performing any other actions.
        """
        grids = self.grids
        if name in grids:
            return grids[name]
        else:
            # new_grid() also modifies the /hook_soln/ state
            grid = self.hook_soln.new_grid(name, *self.dimensions)
            # Allocate memory
            self.hook_soln.prepare_solution()
            return grid


class YaskGrid(object):

    """
    An implementation of an array that behaves similarly to a ``numpy.ndarray``,
    suitable for the YASK storage layout.

    Subclassing ``numpy.ndarray`` would have led to shadow data copies, because
    of the different storage layout.
    """

    # Force __rOP__ methods (OP={add,mul,...) to get arrays, not scalars, for efficiency
    __array_priority__ = 1000

    def __new__(cls, name, shape, dimensions, dtype, buffer=None):
        """
        Create a new YASK Grid and attach it to a "fake" solution.
        """
        # Init YASK if not initialized already
        init(dimensions, shape, dtype)
        # Only create a YaskGrid if the requested grid is a dense one
        if tuple(i.name for i in dimensions) == YASK.dimensions:
            obj = super(YaskGrid, cls).__new__(cls)
            obj.__init__(name, shape, dimensions, dtype, buffer)
            return obj
        else:
            return None

    def __init__(self, name, shape, dimensions, dtype, buffer=None):
        self.name = name
        self.shape = shape
        self.dimensions = dimensions
        self.dtype = dtype

        self.grid = YASK.setdefault(name)

        # Always init the grid, at least with 0.0
        self[:] = 0.0 if buffer is None else buffer

    def __getitem__(self, index):
        # TODO: ATM, no MPI support.
        start, stop, shape = convert_multislice(as_tuple(index), self.shape)
        if not shape:
            debug("YaskGrid: Getting single entry")
            assert start == stop
            out = self.grid.get_element(*start)
        else:
            debug("YaskGrid: Getting full-array/block via index [%s]" % str(index))
            out = np.empty(shape, self.dtype, 'C')
            self.grid.get_elements_in_slice(out.data, start, stop)
        return out

    def __setitem__(self, index, val):
        # TODO: ATM, no MPI support.
        start, stop, shape = convert_multislice(as_tuple(index), self.shape, 'set')
        if all(i == 1 for i in shape):
            debug("YaskGrid: Setting single entry")
            assert start == stop
            self.grid.set_element(val, *start)
        elif isinstance(val, np.ndarray):
            debug("YaskGrid: Setting full-array/block via index [%s]" % str(index))
            self.grid.set_elements_in_slice(val, start, stop)
        elif all(i == j-1 for i, j in zip(shape, self.shape)):
            debug("YaskGrid: Setting full-array to given scalar via single grid sweep")
            self.grid.set_all_elements_same(val)
        else:
            debug("YaskGrid: Setting block to given scalar via index [%s]" % str(index))
            self.grid.set_elements_in_slice_same(val, start, stop)

    def __getslice__(self, start, stop):
        if stop == sys.maxint:
            stop = None
        return self.__getitem__(slice(start, stop))

    def __setslice__(self, start, stop, val):
        if stop == sys.maxint:
            stop = None
        self.__setitem__(slice(start, stop), val)

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

    @property
    def ndpointer(self):
        # TODO: see corresponding comment in interfaces.py about CMemory
        return self


class YaskRewriter(BasicRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._yaskize(state)
        self._create_elemental_functions(state)

    @dle_pass
    def _yaskize(self, state):
        """
        Create a YASK representation of this Iteration/Expression tree.
        """

        dle_warning("Be patient! The YASK backend is still a WIP")
        dle_warning("This is the YASK AST that the Devito DLE can build")

        for node in state.nodes:
            for tree in retrieve_iteration_tree(node):
                candidate = tree[-1]

                # Set up the YASK solution
                soln = YASK.cfac.new_solution("devito-test-solution")

                # Set up the YASK grids
                grids = FindSymbols(mode='symbolics').visit(candidate)
                grids = [YASK.setdefault(i.name) for i in grids]

                # Perform the translation on an expression basis
                transform = sympy2yask(grids)
                expressions = [e for e in candidate.nodes if e.is_Expression]
                try:
                    for i in expressions:
                        transform(i.expr)
                        dle("Converted %s into YASK format", str(i.expr))
                except:
                    dle_warning("Cannot convert %s into YASK format", str(i.expr))
                    continue

                # Print some useful information to screen about the YASK conversion
                dle("Solution '" + soln.get_name() + "' contains " +
                    str(soln.get_num_grids()) + " grid(s), and " +
                    str(soln.get_num_equations()) + " equation(s).")

                # Provide stuff to YASK-land
                # ==========================
                # Scalar: print(ast.format_simple())
                # AVX2 intrinsics: print soln.format('avx2')
                # AVX2 intrinsics to file (active now)
                soln.write(os.path.join(YASK.path, 'yask_stencil_code.hpp'), 'avx2', True)

                # Set necessary run-time parameters
                soln.set_step_dim("t")
                soln.set_element_bytes(4)

        dle_warning("Falling back to basic DLE optimizations...")

        return {'nodes': state.nodes}


class sympy2yask(object):
    """
    Convert a SymPy expression into a YASK abstract syntax tree.
    """

    def __init__(self, grids):
        self.grids = grids
        self.mapper = {}

    def __call__(self, expr):

        def nary2binary(args, op):
            r = run(args[0])
            return r if len(args) == 1 else op(r, nary2binary(args[1:], op))

        def run(expr):
            if expr.is_Integer:
                return YASK.nfac.new_const_number_node(int(expr))
            elif expr.is_Float:
                return YASK.nfac.new_const_number_node(float(expr))
            elif expr.is_Symbol:
                assert expr in self.mapper
                return self.mapper[expr]
            elif isinstance(expr, Indexed):
                function = expr.base.function
                assert function.name in self.grids
                indices = [int((i.origin if isinstance(i, LoweredDimension) else i) - j)
                           for i, j in zip(expr.indices, function.indices)]
                return self.grids[function.name].new_relative_grid_point(*indices)
            elif expr.is_Add:
                return nary2binary(expr.args, YASK.nfac.new_add_node)
            elif expr.is_Mul:
                return nary2binary(expr.args, YASK.nfac.new_multiply_node)
            elif expr.is_Pow:
                num, den = expr.as_numer_denom()
                if num == 1:
                    return YASK.nfac.new_divide_node(run(num), run(den))
            elif expr.is_Equality:
                if expr.lhs.is_Symbol:
                    assert expr.lhs not in self.mapper
                    self.mapper[expr.lhs] = run(expr.rhs)
                else:
                    return YASK.nfac.new_equation_node(*[run(i) for i in expr.args])
            else:
                dle_warning("Missing handler in Devito-YASK translation")
                raise NotImplementedError

        return run(expr)


# YASK interface

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
    global YASK

    if YASK is not None:
        return

    dle("Initializing YASK...")

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

    # Create a new stencil solution
    soln = cfac.new_solution("Hook")
    soln.set_step_dim("t")
    soln.set_domain_dims(*[str(i) for i in dimensions])  # TODO: YASK only accepts x,y,z

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

    # Finish off by initializing YASK
    YASK = YaskState(cfac, nfac, path, env, shape, dtype, hook_soln)

    dle("YASK backend successfully initialized!")


def _force_exit(emsg):
    """
    Handle fatal errors.
    """
    raise DLEException("YASK Error [%s]. Exiting..." % emsg)


# Generic utility functions

def convert_multislice(multislice, shape, mode='get'):
    assert mode in ['get', 'set']
    multislice = as_tuple(multislice)

    # Convert dimensions
    cstart = []
    cstop = []
    cshape = []
    for i, v in enumerate(multislice):
        if isinstance(v, slice):
            if v.step is not None:
                _force_exit("Unsupported stepping != 1.")
            cstart.append(v.start or 0)
            cstop.append((v.stop or shape[i]) - 1)
            cshape.append(cstop[-1] - cstart[-1] + 1)
        else:
            cstart.append(normalize_index(v if v is not None else 0, shape))
            cstop.append(normalize_index(v if v is not None else (shape[i]-1), shape))
            if mode == 'set':
                cshape.append(1)

    # Remainder (e.g., requesting A[1] and A has shape (3,3))
    nremainder = len(shape) - len(multislice)
    cstart.extend([0]*nremainder)
    cstop.extend([shape[i + j] - 1 for j in range(nremainder)])
    cshape.extend([shape[i + j] for j in range(nremainder)])

    return cstart, cstop, cshape


def normalize_index(index, shape):
    normalized = [i if i >= 0 else j + i for i, j in zip(as_tuple(index), shape)]
    return normalized[0] if len(normalized) == 1 else normalized

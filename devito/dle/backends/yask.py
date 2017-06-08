import os
import sys

import numpy as np
from sympy import Indexed

from devito.compiler import make
from devito.dimension import LoweredDimension
from devito.dle import retrieve_iteration_tree
from devito.dle.backends import BasicRewriter, dle_pass
from devito.exceptions import CompilationError
from devito.logger import dle, dle_warning, error
from devito.visitors import FindSymbols

__all__ = ['YaskRewriter', 'init', 'make_grid']


YASK = None
"""Global state for generation of YASK kernels."""


class YaskState(object):

    def __init__(self, cfac, nfac, path, env, settings, hook_soln):
        """
        Global state to interact with YASK.
        """
        self.cfac = cfac  # YASK compiler factory, to create Solutions
        self.nfac = nfac  # YASK node factory, to create ASTs
        self.path = path  # Generated code dump directory
        self.env = env  # Global environment (e.g., MPI)
        self.settings = settings  # Dimensions, grid sizes, etc.
        self.hook_soln = hook_soln  # "Fake" solution to track YASK grids

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

    def setdefault(self, name, vals=0.0):
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
            # Initialization
            grid.set_all_elements(vals)
            # TODO : return YaskGrid (subclass of NumPy array)
            return grid


class YaskGrid(np.ndarray):

    """
    An implementation of a ``numpy.ndarray`` suitable for the YASK storage layout.

    WIP: Currently, the YASK storage layout is assumed transposed w.r.t. the
         usual row-major format.

    This subclass follows the ``numpy`` rules for subclasses detailed at: ::

        https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """

    def __new__(cls, array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        return np.asarray(array).view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __getitem__(self, index):
        expected_layout = self.transpose()
        return super(YaskGrid, expected_layout).__getitem__(index)

    def __setitem__(self, index, val):
        super(YaskGrid, self).__setitem__(index, val)


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

    # Set global settings and create hook solution
    settings = kfac.new_settings()
    hook_soln = kfac.new_solution(env, settings)
    for dm, ds in zip(hook_soln.get_domain_dim_names(), shape):
        # Set domain size in each dim.
        settings.set_rank_domain_size(dm, ds)
        # Set block size to 64 in z dim and 32 in other dims.
        settings.set_block_size(dm, min(64 if dm == "z" else 32, ds))

    # Simple rank configuration in 1st dim only. # TODO Improve me
    settings.set_num_ranks(hook_soln.get_domain_dim_name(0), env.get_num_ranks())

    # Finish off by initializing YASK
    YASK = YaskState(cfac, nfac, path, env, settings, hook_soln)

    dle("YASK backend successfully initialized!")


def make_grid(name, shape, dimensions, dtype):
    """
    Create a new YASK Grid and attach it to a "fake" solution.
    """
    init(dimensions, shape, dtype)
    return YASK.setdefault(name)


def _force_exit(emsg):
    """
    Handle fatal errors.
    """
    error("Couldn't startup YASK [%s]. Exiting..." % emsg)
    sys.exit(0)

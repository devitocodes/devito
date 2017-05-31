import os

import numpy as np
from sympy import Indexed

from devito.dimension import LoweredDimension
from devito.dle import retrieve_iteration_tree
from devito.dle.backends import BasicRewriter, dle_pass
from devito.logger import dle, dle_warning
from devito.visitors import FindSymbols

try:
    import yask_compiler as yask
    # Factories to interact with YASK
    cfac = yask.yc_factory()
    fac = yask.yc_node_factory()
except ImportError:
    yask = None

__all__ = ['YaskRewriter', 'yaskarray']


class YaskRewriter(BasicRewriter):

    def _pipeline(self, state):
        if yask is None:
            dle_warning("Cannot find YASK. Skipping DLE optimizations...")
            super(YaskRewriter, self)._pipeline(state)
            return
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
                soln = cfac.new_solution("devito-test-solution")

                # Set up the YASK grids
                grids = FindSymbols(mode='symbolics').visit(candidate)
                grids = {g.name: soln.new_grid(g.name, *[str(i) for i in g.indices])
                         for g in grids}

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
                #          Scalar: print(ast.format_simple())
                # AVX2 intrinsics: print soln.format('avx2')
                # AVX2 intrinsics to file (active now)
                path = os.path.join(os.environ.get('YASK_HOME', '.'),
                                    'src', 'kernel', 'gen')
                soln.write(os.path.join(path, 'yask_stencil_code.hpp'), 'avx2', True)

                # Set kernel parameters
                # =====================
                # Vector folding API usage example
                soln.set_fold_len('x', 1)
                soln.set_fold_len('y', 1)
                soln.set_fold_len('z', 8)
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
                return fac.new_const_number_node(int(expr))
            elif expr.is_Float:
                return fac.new_const_number_node(float(expr))
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
                return nary2binary(expr.args, fac.new_add_node)
            elif expr.is_Mul:
                return nary2binary(expr.args, fac.new_multiply_node)
            elif expr.is_Pow:
                num, den = expr.as_numer_denom()
                if num == 1:
                    return fac.new_divide_node(run(num), run(den))
            elif expr.is_Equality:
                if expr.lhs.is_Symbol:
                    assert expr.lhs not in self.mapper
                    self.mapper[expr.lhs] = run(expr.rhs)
                else:
                    return fac.new_equation_node(*[run(i) for i in expr.args])
            else:
                dle_warning("Missing handler in Devito-YASK translation")
                raise NotImplementedError

        return run(expr)


class yaskarray(np.ndarray):

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
        if obj is None: return

    def __getitem__(self, index):
        expected_layout = self.transpose()
        return super(yaskarray, expected_layout).__getitem__(index)

    def __setitem__(self, index, val):
        super(yaskarray, self).__setitem__(index, val)

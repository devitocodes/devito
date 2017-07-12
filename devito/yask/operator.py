from __future__ import absolute_import

import os

from sympy import Indexed

from devito.dimension import LoweredDimension
from devito.dle import retrieve_iteration_tree
from devito.logger import dle, dle_warning
from devito.operator import OperatorRunnable

from devito.yask.kernel import YASK

__all__ = ['Operator']


class Operator(OperatorRunnable):

    """
    A special :class:`OperatorCore` to JIT-compile and run operators through YASK.
    """

    def __init__(self, expressions, **kwargs):
        kwargs['dle'] = 'basic'
        super(Operator, self).__init__(expressions, **kwargs)

    def _specialize(self, nodes, elemental_functions):
        """
        Create a YASK representation of this Iteration/Expression tree.
        """

        dle_warning("Be patient! The YASK backend is still a WIP")
        dle_warning("This is the YASK AST that the Devito DLE can build")

        for node in nodes + elemental_functions:
            for tree in retrieve_iteration_tree(node):
                candidate = tree[-1]

                # Set up the YASK solution
                soln = YASK.cfac.new_solution("devito-test-solution")

                # Perform the translation on an expression basis
                transform = sympy2yask(soln)
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
                soln.set_step_dim_name(YASK.time_dimension)
                soln.set_domain_dim_names(YASK.space_dimensions)
                soln.set_element_bytes(4)

        dle_warning("Falling back to basic DLE optimizations...")

        # TODO: need to update nodes and elemental_functions

        return nodes, elemental_functions


class sympy2yask(object):
    """
    Convert a SymPy expression into a YASK abstract syntax tree and create any
    necessay YASK grids.
    """

    def __init__(self, soln):
        self.soln = soln
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
                name = function.name
                if name not in YASK.grids:
                    function.data  # Create uninitialized grid (i.e., 0.0 everywhere)
                dimensions = YASK.grids[name].get_dim_names()
                grid = self.mapper.setdefault(name, self.soln.new_grid(name, dimensions))
                indices = [int((i.origin if isinstance(i, LoweredDimension) else i) - j)
                           for i, j in zip(expr.indices, function.indices)]
                return grid.new_relative_grid_point(*indices)
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

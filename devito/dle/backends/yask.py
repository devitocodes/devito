from sympy import Indexed

from devito.dimension import LoweredDimension
from devito.dle import retrieve_iteration_tree
from devito.dle.backends import BasicRewriter, dle_pass
from devito.logger import dle_warning
from devito.visitors import FindSymbols

try:
    import yask_compiler as yask
    # Factories to interact with YASK
    cfac = yask.yask_compiler_factory()
    fac = yask.node_factory()
except ImportError:
    yask = None


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
                soln = cfac.new_stencil_solution("solution")

                # Set up the YASK grids
                grids = FindSymbols(mode='symbolics').visit(candidate)
                grids = {g.name: soln.new_grid(g.name, *[str(i) for i in g.indices])
                         for g in grids}

                # Vector folding API usage example
                soln.set_fold_len('x', 1)
                soln.set_fold_len('y', 1)
                soln.set_fold_len('z', 8)

                # Perform the translation on an expression basis
                transformer = sympy2yask(grids)
                expressions = [e for e in candidate.nodes if e.is_Expression]
                # yaskASTs = [transformer(e.stencil) for e in expressions]
                for i in expressions:
                    try:
                        ast = transformer(i.stencil)
                        # Scalar
                        print(ast.format_simple())

                        # AVX2 intrinsics
                        # print soln.format('avx2')

                        # AVX2 intrinsics to file
                        # import os
                        # path = os.path.join(os.environ.get('YASK_HOME', '.'), 'src')
                        # soln.write(os.path.join(path, 'stencil_code.hpp'), 'avx2')
                    except:
                        pass

        dle_warning("Falling back to basic DLE optimizations...")

        return {'nodes': state.nodes}


class sympy2yask(object):
    """
    Convert a SymPy expression into a YASK abstract syntax tree.
    """

    def __init__(self, grids):
        self.grids = grids

    def __call__(self, expr):

        def nary2binary(args, op):
            r = run(args[0])
            return r if len(args) == 1 else op(r, nary2binary(args[1:], op))

        def run(expr):
            if expr.is_Integer:
                return fac.new_const_number_node(int(expr))
            elif expr.is_Float:
                return fac.new_const_number_node(float(expr))
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
                return fac.new_equation_node(*[run(i) for i in expr.args])
            else:
                dle_warning("Missing handler in Devito-YASK translation")
                raise NotImplementedError

        return run(expr)

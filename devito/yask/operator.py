from __future__ import absolute_import

from sympy import Indexed

from devito import configuration
from devito.compiler import jit_compile
from devito.dimension import LoweredDimension
from devito.dle import filter_iterations, retrieve_iteration_tree
from devito.interfaces import Object
from devito.nodes import FunCall
from devito.logger import yask as log, yask_warning as warning
from devito.operator import OperatorRunnable, FunMeta
from devito.visitors import FindSymbols, Transformer

from devito.yask import cfac, nfac, ofac, namespace, exit
from devito.yask.wrappers import yask_context

__all__ = ['Operator']


class Operator(OperatorRunnable):

    """
    A special :class:`OperatorCore` to JIT-compile and run operators through YASK.
    """

    _default_headers = OperatorRunnable._default_headers
    _default_headers += ['#define restrict __restrict']
    _default_includes = OperatorRunnable._default_includes + ['yask_kernel_api.hpp']

    def __init__(self, expressions, **kwargs):
        kwargs['dle'] = 'noop'
        super(Operator, self).__init__(expressions, **kwargs)
        # Each YASK Operator needs to have its own compiler (hence the copy()
        # below) because Operator-specific shared object will be added to the
        # list of linked libraries
        self._compiler = configuration['compiler-yask'].copy()

    def _specialize(self, nodes, parameters):
        """
        Create a YASK representation of this Iteration/Expression tree.

        ``parameters`` is modified in-place adding YASK-related arguments.
        """

        log("Specializing a Devito Operator for YASK...")

        # Set up the YASK solution
        ycsoln = cfac.new_solution(namespace['kernel-real'])

        # Silence YASK
        self.yask_compiler_output = ofac.new_string_output()
        ycsoln.set_debug_output(self.yask_compiler_output)

        trees = retrieve_iteration_tree(nodes)
        if len(trees) > 1:
            exit("Currently unable to handle Operators w/ more than 1 loop nests")
        tree = trees[0]
        candidate = tree[-1]

        expressions = [e for e in candidate.nodes if e.is_Expression]
        functions = [i for i in FindSymbols().visit(candidate) if i.is_TimeData]
        keys = set([(i.indices, i.shape, i.dtype, i.space_order) for i in functions])
        if len(keys) > 1:
            exit("Currently unable to handle Operators w/ heterogeneous grids")
        self.context = yask_context(*keys.pop())

        # Perform the translation on an expression basis
        transform = sympy2yask(self.context, ycsoln)
        try:
            for i in expressions:
                ast = transform(i.expr)
                log("Converted %s into YASK AST [%s]", str(i.expr), ast.format_simple())
        except:
            exit("Couldn't convert %s into YASK format" % str(i.expr))

        # Print some useful information about the newly constructed Yask solution
        log("Solution '%s' contains %d grid(s) and %d equation(s)." %
            (ycsoln.get_name(), ycsoln.get_num_grids(), ycsoln.get_num_equations()))

        # Set necessary run-time parameters
        ycsoln.set_step_dim_name(self.context.time_dimension)
        ycsoln.set_domain_dim_names(self.context.space_dimensions)
        ycsoln.set_element_bytes(4)

        # JIT-compile the newly-created YASK kernel
        self.ksoln = self.context.make_solution(ycsoln)

        # TODO: need to update nodes
        key = lambda i: i.dim.name == self.ksoln.space_dimensions[0]
        root = filter_iterations(tree, key=key, stop='asap')
        if len(root) != 1:
            exit("Couldn't find the root space loop within:\n%s" % str(tree[0]))
        root = root[0]

        funcall = '(*soln)->run_solution'
        processed = Transformer({root: FunCall(funcall, 'time')}).visit(nodes)

        # Track this is an external function call
        self.func_table[funcall] = FunMeta(None, False)

        # Add the Yask solution to the parameters list
        parameters.append(Object('soln', namespace['type-solution'],
                                 self.ksoln.rawpointer))

        log("Specialization successfully performed!")

        return processed

    def apply(self, **kwargs):
        # Build the arguments list to invoke the kernel function
        arguments, dim_sizes = self.arguments(**kwargs)

        # Share the grids from the hook solution
        for kgrid in self.ksoln.grids:
            hgrid = self.context.grids[kgrid.get_name()]
            kgrid.share_storage(hgrid)
            log("Shared storage from hook grid <%s>" % hgrid.get_name())

        # Print some info about the solution.
        log("Stencil-solution '%s':" % self.ksoln.name)
        log("  Step dimension: %s" % self.context.time_dimension)
        log("  Domain dimensions: %s" % str(self.context.space_dimensions))
        log("  Grids:")
        for grid in self.ksoln.grids:
            pad = str([grid.get_pad_size(i) for i in self.context.space_dimensions])
            log("    %s%s, pad=%s" % (grid.get_name(), str(grid.get_dim_names()), pad))

        # Required by YASK before running any stencils
        self.ksoln.prepare()

        ntimesteps = dim_sizes[self.context.time_dimension]
        # TODO: will be able to run through either YASK or Devito
        if True:
            log("Running YASK Operator through YASK...")
            self.ksoln.run(ntimesteps)
        else:
            log("Running YASK Operator through Devito...")
            from IPython import embed; embed()
            self.cfunction(*list(arguments.values()))
        log("YASK Operator successfully run!")

    @property
    def compile(self):
        """
        JIT-compile the C code generated by the Operator.

        It is ensured that JIT compilation will only be performed once per
        :class:`Operator`, reagardless of how many times this method is invoked.

        :returns: The file name of the JIT-compiled function.
        """
        if self._lib is None:
            # No need to recompile if a shared object has already been loaded.
            self._compiler.libraries.append(self.ksoln.soname)
            return jit_compile(self.ccode, self._compiler)
        else:
            return self._lib.name


class sympy2yask(object):
    """
    Convert a SymPy expression into a YASK abstract syntax tree and create any
    necessay YASK grids.
    """

    def __init__(self, context, soln):
        self.context = context
        self.soln = soln
        self.mapper = {}

    def __call__(self, expr):

        def nary2binary(args, op):
            r = run(args[0])
            return r if len(args) == 1 else op(r, nary2binary(args[1:], op))

        def run(expr):
            if expr.is_Integer:
                return nfac.new_const_number_node(int(expr))
            elif expr.is_Float:
                return nfac.new_const_number_node(float(expr))
            elif expr.is_Symbol:
                assert expr in self.mapper
                return self.mapper[expr]
            elif isinstance(expr, Indexed):
                function = expr.base.function
                name = function.name
                if name not in self.context.grids:
                    function.data  # Create uninitialized grid (i.e., 0.0 everywhere)
                if name not in self.mapper:
                    dimensions = self.context.grids[name].get_dim_names()
                    self.mapper[name] = self.soln.new_grid(name, dimensions)
                indices = [int((i.origin if isinstance(i, LoweredDimension) else i) - j)
                           for i, j in zip(expr.indices, function.indices)]
                return self.mapper[name].new_relative_grid_point(*indices)
            elif expr.is_Add:
                return nary2binary(expr.args, nfac.new_add_node)
            elif expr.is_Mul:
                return nary2binary(expr.args, nfac.new_multiply_node)
            elif expr.is_Pow:
                num, den = expr.as_numer_denom()
                if num == 1:
                    return nfac.new_divide_node(run(num), run(den))
            elif expr.is_Equality:
                if expr.lhs.is_Symbol:
                    assert expr.lhs not in self.mapper
                    self.mapper[expr.lhs] = run(expr.rhs)
                else:
                    return nfac.new_equation_node(*[run(i) for i in expr.args])
            else:
                warning("Missing handler in Devito-YASK translation")
                raise NotImplementedError

        return run(expr)

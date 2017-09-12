from __future__ import absolute_import

import cgen as c
from sympy import Indexed

from devito.cgen_utils import ccode
from devito.compiler import jit_compile
from devito.dimension import LoweredDimension
from devito.dle import filter_iterations, retrieve_iteration_tree
from devito.interfaces import Object
from devito.logger import yask as log, yask_warning as warning
from devito.nodes import Element
from devito.operator import OperatorRunnable, FunMeta
from devito.tools import flatten
from devito.visitors import IsPerfectIteration, Transformer

from devito.yask import cfac, nfac, ofac, namespace, exit, yask_configuration
from devito.yask.utils import make_grid_accesses, make_sharedptr_funcall
from devito.yask.wrappers import (YaskGrid, YaskNullContext, YaskNullSolution,
                                  yask_context)

__all__ = ['Operator']


class Operator(OperatorRunnable):

    """
    A special :class:`OperatorCore` to JIT-compile and run operators through YASK.
    """

    _default_headers = OperatorRunnable._default_headers
    _default_headers += ['#define restrict __restrict']
    _default_includes = OperatorRunnable._default_includes + ['yask_kernel_api.hpp']

    def __init__(self, expressions, **kwargs):
        kwargs['dle'] = 'basic'
        super(Operator, self).__init__(expressions, **kwargs)
        # Each YASK Operator needs to have its own compiler (hence the copy()
        # below) because Operator-specific shared object will be added to the
        # list of linked libraries
        self._compiler = yask_configuration['compiler'].copy()

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

        # Find offloadable Iteration/Expression trees
        offloadable = []
        for tree in retrieve_iteration_tree(nodes):
            parallel = filter_iterations(tree, lambda i: i.is_Parallel)
            if not parallel:
                # Cannot offload non-parallel loops
                continue
            if not (IsPerfectIteration().visit(tree) and
                    all(i.is_Expression for i in tree[-1].nodes)):
                # Don't know how to offload this Iteration/Expression to YASK
                continue
            functions = flatten(i.functions for i in tree[-1].nodes)
            keys = set([(i.indices, i.shape, i.dtype, i.space_order) for i in functions
                        if i.is_TimeData])
            if len(keys) == 0:
                continue
            elif len(keys) > 1:
                exit("Cannot handle Operators w/ heterogeneous grids")
            dimensions, shape, dtype, space_order = keys.pop()
            if len(dimensions) == len(tree) and\
                    all(i.dim == j for i, j in zip(tree, dimensions)):
                # Detected a "full" Iteration/Expression tree (over both
                # time and space dimensions)
                offloadable.append((tree, dimensions, shape, dtype, space_order))

        # Construct YASK ASTs given Devito expressions. New grids may be allocated.
        if len(offloadable) == 0:
            # No offloadable trees found
            self.context = YaskNullContext()
            self.ksoln = YaskNullSolution()
            processed = nodes
            log("No offloadable trees found")
        elif len(offloadable) == 1:
            # Found *the* offloadable tree for this Operator
            tree, dimensions, shape, dtype, space_order = offloadable[0]
            self.context = yask_context(dimensions, shape, dtype, space_order)

            transform = sympy2yask(self.context, ycsoln)
            try:
                for i in tree[-1].nodes:
                    transform(i.expr)

                funcall = make_sharedptr_funcall(namespace['code-soln-run'], ['time'],
                                                 namespace['code-soln-name'])
                funcall = Element(c.Statement(ccode(funcall)))
                processed = Transformer({tree[1]: funcall}).visit(nodes)

                # Track this is an external function call
                self.func_table[namespace['code-soln-run']] = FunMeta(None, False)

                # Set necessary run-time parameters
                ycsoln.set_step_dim_name(self.context.time_dimension)
                ycsoln.set_domain_dim_names(self.context.space_dimensions)
                ycsoln.set_element_bytes(4)

                # JIT-compile the newly-created YASK kernel
                self.ksoln = self.context.make_solution(ycsoln)

                # Now we must drop a pointer to the YASK solution down to C-land
                parameters.append(Object(namespace['code-soln-name'],
                                         namespace['type-solution'],
                                         self.ksoln.rawpointer))

                # Print some useful information about the newly constructed solution
                log("Solution '%s' contains %d grid(s) and %d equation(s)." %
                    (ycsoln.get_name(), ycsoln.get_num_grids(),
                     ycsoln.get_num_equations()))
            except:
                self.ksoln = YaskNullSolution()
                processed = nodes
                log("Unable to offload a candidate tree.")
        else:
            exit("Found more than one offloadable trees in a single Operator")

        # Some Iteration/Expression trees are not offloaded to YASK and may
        # require further processing to be executed in YASK, due to the differences
        # in storage layout employed by Devito and YASK
        processed = make_grid_accesses(processed)

        # Update the parameters list adding all necessary YASK grids
        for i in list(parameters):
            try:
                if i.is_SymbolicData and isinstance(i.data, YaskGrid):
                    parameters.append(Object(namespace['code-grid-name'](i.name),
                                             namespace['type-grid'],
                                             i.data.rawpointer))
            except AttributeError:
                # Ignore e.g. Dimensions
                pass

        log("Specialization successfully performed!")

        return processed

    def arguments(self, **kwargs):
        # The user has the illusion to provide plain data objects to the
        # generated kernels, but what we will actually also drop in are
        # pointers to the wrapped YASK grids
        for i in self.parameters:
            obj = kwargs.get(i.name)
            if i.is_PtrArgument and obj is not None:
                assert(i.verify(obj.data))
        return super(Operator, self).arguments(**kwargs)

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

        if yask_configuration['python-exec']:
            log("Running YASK Operator through YASK...")
            self.ksoln.run(dim_sizes[self.context.time_dimension])
        else:
            log("Running YASK Operator through Devito...")
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
            if not isinstance(self.ksoln, YaskNullSolution):
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

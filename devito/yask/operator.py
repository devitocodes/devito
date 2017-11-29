from __future__ import absolute_import

import cgen as c
import numpy as np
from sympy import Indexed

from devito.cgen_utils import ccode
from devito.compiler import jit_compile
from devito.dimension import LoweredDimension
from devito.types import Object
from devito.logger import yask as log, yask_warning as warning
from devito.ir.iet import (Element, IsPerfectIteration, Transformer,
                           filter_iterations, retrieve_iteration_tree)
from devito.operator import OperatorRunnable, FunMeta
from devito.tools import flatten

from devito.yask import nfac, namespace, exit, configuration
from devito.yask.utils import make_grid_accesses, make_sharedptr_funcall, rawpointer
from devito.yask.wrappers import YaskGridConst, YaskNullContext, YaskNullKernel, contexts

__all__ = ['Operator']


class Operator(OperatorRunnable):

    """
    A special :class:`OperatorCore` to JIT-compile and run operators through YASK.
    """

    _default_headers = OperatorRunnable._default_headers
    _default_headers += ['#define restrict __restrict']
    _default_includes = OperatorRunnable._default_includes + ['yask_kernel_api.hpp']

    def __init__(self, expressions, **kwargs):
        kwargs['dle'] = ('denormals',) + (('openmp',) if configuration['openmp'] else ())
        super(Operator, self).__init__(expressions, **kwargs)
        # Each YASK Operator needs to have its own compiler (hence the copy()
        # below) because Operator-specific shared object will be added to the
        # list of linked libraries
        self._compiler = configuration.yask['compiler'].copy()

    def _specialize(self, nodes, parameters):
        """
        Create a YASK representation of this Iteration/Expression tree.

        ``parameters`` is modified in-place adding YASK-related arguments.
        """
        log("Specializing a Devito Operator for YASK...")

        self.context = YaskNullContext()
        self.yk_soln = YaskNullKernel()
        local_grids = []

        offloadable = find_offloadable_trees(nodes)
        if len(offloadable) == 0:
            log("No offloadable trees found")
        elif len(offloadable) == 1:
            tree, grid, dtype = offloadable[0]
            self.context = contexts.fetch(grid, dtype)

            # Create a YASK compiler solution for this Operator
            yc_soln = self.context.make_yc_solution(namespace['jit-yc-soln'])

            transform = sympy2yask(self.context, yc_soln)
            try:
                for i in tree[-1].nodes:
                    transform(i.expr)

                funcall = make_sharedptr_funcall(namespace['code-soln-run'], ['time'],
                                                 namespace['code-soln-name'])
                funcall = Element(c.Statement(ccode(funcall)))
                nodes = Transformer({tree[1]: funcall}).visit(nodes)

                # Track /funcall/ as an external function call
                self.func_table[namespace['code-soln-run']] = FunMeta(None, False)

                # JIT-compile the newly-created YASK kernel
                local_grids += [i for i in transform.mapper if i.is_Array]
                self.yk_soln = self.context.make_yk_solution(namespace['jit-yk-soln'],
                                                             yc_soln, local_grids)

                # Now we must drop a pointer to the YASK solution down to C-land
                parameters.append(Object(namespace['code-soln-name'],
                                         namespace['type-solution'],
                                         self.yk_soln.rawpointer))

                # Print some useful information about the newly constructed solution
                log("Solution '%s' contains %d grid(s) and %d equation(s)." %
                    (yc_soln.get_name(), yc_soln.get_num_grids(),
                     yc_soln.get_num_equations()))
            except:
                log("Unable to offload a candidate tree.")
        else:
            exit("Found more than one offloadable trees in a single Operator")

        # Some Iteration/Expression trees are not offloaded to YASK and may
        # require further processing to be executed in YASK, due to the differences
        # in storage layout employed by Devito and YASK
        nodes = make_grid_accesses(nodes)

        # Update the parameters list adding all necessary YASK grids
        for i in list(parameters) + local_grids:
            try:
                if i.from_YASK:
                    parameters.append(Object(namespace['code-grid-name'](i.name),
                                             namespace['type-grid']))
            except AttributeError:
                # Ignore e.g. Dimensions
                pass

        log("Specialization successfully performed!")

        return nodes

    def arguments(self, **kwargs):
        mapper = {i.name: i for i in self.parameters}
        local_grids_mapper = {namespace['code-grid-name'](k): v
                              for k, v in self.yk_soln.local_grids.items()}

        # The user has the illusion to provide plain data objects to the
        # generated kernels, but what we actually need and thus going to
        # provide are pointers to the wrapped YASK grids.
        toshare = {}
        for i in self.parameters:
            grid_arg = mapper.get(namespace['code-grid-name'](i.name))
            if grid_arg is not None:
                assert i.provider.from_YASK is True
                obj = kwargs.get(i.name, i.provider)
                # Get the associated YaskGrid wrapper (scalars are a special case)
                if np.isscalar(obj):
                    wrapper = YaskGridConst(obj)
                    toshare[i.provider] = wrapper
                else:
                    wrapper = obj.data
                    toshare[obj] = wrapper
                # Add C-level pointer to the YASK grids
                assert grid_arg.verify(wrapper.rawpointer)
            elif i.name in local_grids_mapper:
                # Add C-level pointer to the temporary YASK grids
                assert i.verify(rawpointer(local_grids_mapper[i.name]))

        return super(Operator, self).arguments(**kwargs), toshare

    def apply(self, **kwargs):
        # Build the arguments list to invoke the kernel function
        arguments, toshare = self.arguments(**kwargs)

        log("Running YASK Operator through Devito...")
        self.yk_soln.run(self.cfunction, arguments, toshare)
        log("YASK Operator successfully run!")

        # Output summary of performance achieved
        return self._profile_output(arguments)

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
            if not isinstance(self.yk_soln, YaskNullKernel):
                self._compiler.libraries.append(self.yk_soln.soname)
            return jit_compile(self.ccode, self._compiler)
        else:
            return self._lib.name


class sympy2yask(object):
    """
    Convert a SymPy expression into a YASK abstract syntax tree and create any
    necessay YASK grids.
    """

    def __init__(self, context, yc_soln):
        self.context = context
        self.yc_soln = yc_soln
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
            elif expr.is_Rational:
                a, b = expr.as_numer_denom()
                return nfac.new_const_number_node(float(a)/float(b))
            elif expr.is_Symbol:
                function = expr.base.function
                if function.is_Constant:
                    if function not in self.mapper:
                        self.mapper[function] = self.yc_soln.new_grid(function.name, [])
                    return self.mapper[function].new_relative_grid_point([])
                else:
                    # A DSE-generated temporary, which must have already been
                    # encountered as a LHS of a previous expression
                    assert function in self.mapper
                    return self.mapper[function]
            elif isinstance(expr, Indexed):
                function = expr.base.function
                if function not in self.mapper:
                    if function.is_TimeFunction:
                        dimensions = [nfac.new_step_index(function.indices[0].name)]
                        dimensions += [nfac.new_domain_index(i.name)
                                       for i in function.indices[1:]]
                    else:
                        dimensions = [nfac.new_domain_index(i.name)
                                      for i in function.indices]
                    self.mapper[function] = self.yc_soln.new_grid(function.name,
                                                                  dimensions)
                indices = [int((i.origin if isinstance(i, LoweredDimension) else i) - j)
                           for i, j in zip(expr.indices, function.indices)]
                return self.mapper[function].new_relative_grid_point(indices)
            elif expr.is_Add:
                return nary2binary(expr.args, nfac.new_add_node)
            elif expr.is_Mul:
                return nary2binary(expr.args, nfac.new_multiply_node)
            elif expr.is_Pow:
                base, exp = expr.as_base_exp()
                if not exp.is_integer:
                    warning("non-integer powers unsupported in Devito-YASK translation")
                    raise NotImplementedError

                if int(exp) < 0:
                    num, den = expr.as_numer_denom()
                    return nfac.new_divide_node(run(num), run(den))
                elif int(exp) >= 1:
                    return nary2binary([base] * exp, nfac.new_multiply_node)
                else:
                    warning("0-power found in Devito-YASK translation? setting to 1")
                    return nfac.new_const_number_node(1)
            elif expr.is_Equality:
                if expr.lhs.is_Symbol:
                    function = expr.lhs.base.function
                    assert function not in self.mapper
                    self.mapper[function] = run(expr.rhs)
                else:
                    return nfac.new_equation_node(*[run(i) for i in expr.args])
            else:
                warning("Missing handler in Devito-YASK translation")
                raise NotImplementedError

        return run(expr)


def find_offloadable_trees(nodes):
    """
    Return the trees within ``nodes`` that can be computed by YASK.

    A tree is "offloadable to YASK" if it is embedded in a time stepping loop
    *and* all of the grids accessed by the enclosed equations are homogeneous
    (i.e., same dimensions and data type).
    """
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
        keys = set((i.grid, i.dtype) for i in functions if i.is_TimeFunction)
        if len(keys) == 0:
            continue
        elif len(keys) > 1:
            exit("Cannot handle Operators w/ heterogeneous grids")
        grid, dtype = keys.pop()
        # Is this a "complete" tree iterating over the entire grid?
        dims = [i.dim for i in tree]
        if all(i in dims for i in grid.dimensions) and\
                any(j in dims for j in [grid.time_dim, grid.stepping_dim]):
            offloadable.append((tree, grid, dtype))
    return offloadable

from __future__ import absolute_import

import cgen as c
import numpy as np
from sympy import Indexed

from devito.cgen_utils import ccode
from devito.compiler import jit_compile
from devito.dimension import LoweredDimension
from devito.logger import yask as log, yask_warning as warning
from devito.ir.equations import LoweredEq
from devito.ir.iet import (Element, List, PointerCast, MetaCall, IsPerfectIteration,
                           Transformer, filter_iterations, retrieve_iteration_tree)
from devito.operator import OperatorRunnable
from devito.tools import flatten
from devito.types import Object

from devito.yask import nfac, namespace, exit, configuration
from devito.yask.data import DataScalar
from devito.yask.utils import make_grid_accesses, make_sharedptr_funcall, rawpointer
from devito.yask.wrappers import YaskNullContext, YaskNullKernel, contexts
from devito.yask.types import YaskGridObject

__all__ = ['Operator']


class Operator(OperatorRunnable):

    """
    A special :class:`OperatorCore` to JIT-compile and run operators through YASK.
    """

    _default_headers = OperatorRunnable._default_headers
    _default_headers += ['#define restrict __restrict']
    _default_includes = OperatorRunnable._default_includes + ['yask_kernel_api.hpp']

    def __init__(self, expressions, **kwargs):
        super(Operator, self).__init__(expressions, **kwargs)
        # Each YASK Operator needs to have its own compiler (hence the copy()
        # below) because Operator-specific shared object will be added to the
        # list of linked libraries
        self._compiler = configuration.yask['compiler'].copy()

    def _specialize_exprs(self, expressions):
        expressions = super(Operator, self)._specialize_exprs(expressions)
        # No matter whether offloading will occur or not, all YASK grids accept
        # negative indices when using the get/set_element_* methods (up to the
        # padding extent), so the OOB-relative data space should be adjusted
        return [LoweredEq(e, e.ispace,
                          e.dspace.zero([d for d in e.dimensions if d.is_Space]),
                          e.reads, e.writes)
                for e in expressions]

    def _specialize_iet(self, nodes):
        """Transform the Iteration/Expression tree to offload the computation of
        one or more loop nests onto YASK. This involves calling the YASK compiler
        to generate YASK code. Such YASK code is then called from within the
        transformed Iteration/Expression tree."""
        log("Specializing a Devito Operator for YASK...")

        self.context = YaskNullContext()
        self.yk_soln = YaskNullKernel()

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
                self.func_table[namespace['code-soln-run']] = MetaCall(None, False)

                # JIT-compile the newly-created YASK kernel
                local_grids = [i for i in transform.mapper if i.is_Array]
                self.yk_soln = self.context.make_yk_solution(namespace['jit-yk-soln'],
                                                             yc_soln, local_grids)

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

        log("Specialization successfully performed!")

        return nodes

    def _build_parameters(self, nodes):
        parameters = super(Operator, self)._build_parameters(nodes)
        # Add parameters "disappeared" due to offloading
        parameters += tuple(i for i in self.input if i not in parameters)
        return parameters

    def _build_casts(self, nodes):
        nodes = super(Operator, self)._build_casts(nodes)

        # Add YASK solution pointer for use in C-land
        soln_obj = Object(namespace['code-soln-name'], namespace['type-solution'])

        # Add YASK user and local grids pointers for use in C-land
        grid_objs = [YaskGridObject(i.name) for i in self.input if i.from_YASK]
        grid_objs.extend([YaskGridObject(i) for i in self.yk_soln.local_grids])

        # Build pointer casts
        casts = [PointerCast(soln_obj)] + [PointerCast(i) for i in grid_objs]

        return List(body=casts + [nodes])

    def arguments(self, **kwargs):
        args = {}
        # Add in solution pointer
        args[namespace['code-soln-name']] = self.yk_soln.rawpointer
        # Add in local grids pointers
        for k, v in self.yk_soln.local_grids.items():
            args[namespace['code-grid-name'](k)] = rawpointer(v)
        return super(Operator, self).arguments(backend=args, **kwargs)

    def apply(self, **kwargs):
        # Build the arguments list to invoke the kernel function
        args = self.arguments(**kwargs)

        # Map default Functions to runtime Functions; will be used for "grid sharing"
        toshare = {}
        for i in self.input:
            v = kwargs.get(i.name, i)
            if np.isscalar(v):
                toshare[i] = DataScalar(v)
            elif i.from_YASK and (i.is_Constant or i.is_Function):
                toshare[v] = v.data

        log("Running YASK Operator through Devito...")
        arg_values = [args[p.name] for p in self.parameters]
        self.yk_soln.run(self.cfunction, arg_values, toshare)
        log("YASK Operator successfully run!")

        # Output summary of performance achieved
        return self._profile_output(args)

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
        if not (IsPerfectIteration().visit(parallel[0]) and
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

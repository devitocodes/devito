from __future__ import absolute_import

import cgen as c
import numpy as np

from devito.cgen_utils import ccode
from devito.compiler import jit_compile
from devito.logger import yask as log
from devito.ir.equations import LoweredEq
from devito.ir.iet import (Element, List, PointerCast, MetaCall, Transformer,
                           retrieve_iteration_tree)
from devito.operator import OperatorRunnable
from devito.tools import flatten
from devito.types import Object

from devito.yask import configuration
from devito.yask.data import DataScalar
from devito.yask.utils import (make_grid_accesses, make_sharedptr_funcall, rawpointer,
                               namespace)
from devito.yask.wrappers import YaskNullContext, YaskNullKernel, contexts
from devito.yask.transformer import yaskizer
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

    def _specialize_iet(self, iet):
        """
        Transform the Iteration/Expression tree to offload the computation of
        one or more loop nests onto YASK. This involves calling the YASK compiler
        to generate YASK code. Such YASK code is then called from within the
        transformed Iteration/Expression tree.
        """
        log("Specializing a Devito Operator for YASK...")

        offloadable = find_offloadable_trees(iet)
        if len(offloadable) == 0:
            self.context = YaskNullContext()
            self.yk_soln = YaskNullKernel()

            log("No offloadable trees found")
        else:
            root, grid, dtype = offloadable[0]
            self.context = contexts.fetch(grid, dtype)

            # Create a YASK compiler solution for this Operator
            yc_soln = self.context.make_yc_solution(namespace['jit-yc-soln'])

            try:
                mapper = yaskizer(root, yc_soln)

                funcall = make_sharedptr_funcall(namespace['code-soln-run'], ['time'],
                                                 namespace['code-soln-name'])
                funcall = Element(c.Statement(ccode(funcall)))
                iet = Transformer({root: funcall}).visit(iet)

                # Track /funcall/ as an external function call
                self.func_table[namespace['code-soln-run']] = MetaCall(None, False)

                # JIT-compile the newly-created YASK kernel
                local_grids = [i for i in mapper if i.is_Array]
                self.yk_soln = self.context.make_yk_solution(namespace['jit-yk-soln'],
                                                             yc_soln, local_grids)

                # Print some useful information about the newly constructed solution
                log("Solution '%s' contains %d grid(s) and %d equation(s)." %
                    (yc_soln.get_name(), yc_soln.get_num_grids(),
                     yc_soln.get_num_equations()))
            except:
                log("Unable to offload a candidate tree.")

        # Some Iteration/Expression trees are not offloaded to YASK and may
        # require further processing to be executed in YASK, due to the differences
        # in storage layout employed by Devito and YASK
        iet = make_grid_accesses(iet)

        log("Specialization successfully performed!")

        return iet

    def _build_parameters(self, iet):
        parameters = super(Operator, self)._build_parameters(iet)
        # Add parameters "disappeared" due to offloading
        parameters += tuple(i for i in self.input if i not in parameters)
        return parameters

    def _build_casts(self, iet):
        iet = super(Operator, self)._build_casts(iet)

        # Add YASK solution pointer for use in C-land
        soln_obj = Object(namespace['code-soln-name'], namespace['type-solution'])

        # Add YASK user and local grids pointers for use in C-land
        grid_objs = [YaskGridObject(i.name) for i in self.input if i.from_YASK]
        grid_objs.extend([YaskGridObject(i) for i in self.yk_soln.local_grids])

        # Build pointer casts
        casts = [PointerCast(soln_obj)] + [PointerCast(i) for i in grid_objs]

        return List(body=casts + [iet])

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


def find_offloadable_trees(iet):
    """
    Return the trees within ``iet`` that can be computed by YASK.

    A tree is "offloadable to YASK" if it is embedded in a time stepping loop
    *and* all of the grids accessed by the enclosed equations are homogeneous
    (i.e., same dimensions and data type).
    """
    offloadable = []
    for tree in retrieve_iteration_tree(iet):
        # The outermost iteration must be over time and it must
        # nest at least one iteration
        if len(tree) <= 1:
            continue
        time_iteration = tree[0]
        if not time_iteration.dim.is_Time:
            continue
        grid_iterations = tree[1:]
        if not all(i.is_Affine for i in grid_iterations):
            # Non-affine array accesses unsupported by YASK
            continue
        bundle = tree[-1].nodes[0]
        if len(tree.inner.nodes) > 1 or not bundle.is_ExpressionBundle:
            # Illegal nest
            continue
        functions = flatten(i.functions for i in bundle.exprs)
        keys = set((i.grid, i.dtype) for i in functions if i.is_TimeFunction)
        if len(keys) == 0:
            # No Function found in this tree?
            continue
        assert len(keys) == 1
        grid, dtype = keys.pop()
        # Does this tree iterate over a convex section of the grid?
        if not all(i.dim._defines | set(grid.dimensions) for i in grid_iterations):
            continue
        offloadable.append((grid_iterations[0], grid, dtype))
    return offloadable

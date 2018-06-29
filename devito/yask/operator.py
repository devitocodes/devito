from __future__ import absolute_import

from collections import namedtuple

import cgen as c
import numpy as np

from devito.cgen_utils import ccode
from devito.compiler import jit_compile
from devito.logger import yask as log
from devito.ir.equations import LoweredEq
from devito.ir.iet import (Element, List, PointerCast, MetaCall, Transformer,
                           retrieve_iteration_tree)
from devito.ir.support import align_accesses
from devito.operator import OperatorRunnable
from devito.tools import ReducerMap, Signer, flatten
from devito.types import Object

from devito.yask import configuration
from devito.yask.data import DataScalar
from devito.yask.utils import (make_grid_accesses, make_sharedptr_funcall, rawpointer,
                               namespace)
from devito.yask.wrappers import YaskNullKernel, contexts
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
        # Align data accesses to the computational domain if not a yask.Function
        key = lambda i: i.is_TensorFunction and not i.from_YASK
        expressions = [align_accesses(e, key=key) for e in expressions]

        expressions = super(Operator, self)._specialize_exprs(expressions)

        # No matter whether offloading will occur or not, all YASK grids accept
        # negative indices when using the get/set_element_* methods (up to the
        # padding extent), so the OOB-relative data space should be adjusted
        return [LoweredEq(e, e.ispace,
                          e.dspace.zero([d for d in e.dimensions if d.is_Space]),
                          e.reads, e.writes)
                for e in expressions]

    def _specialize_iet(self, iet, **kwargs):
        """
        Transform the Iteration/Expression tree to offload the computation of
        one or more loop nests onto YASK. This involves calling the YASK compiler
        to generate YASK code. Such YASK code is then called from within the
        transformed Iteration/Expression tree.
        """
        offloadable = find_offloadable_trees(iet)

        if len(offloadable.trees) == 0:
            self.yk_soln = YaskNullKernel()

            log("No offloadable trees found")
        else:
            context = contexts.fetch(offloadable.grid, offloadable.dtype)

            # A unique name for the 'real' compiler and kernel solutions
            name = namespace['jit-soln'](Signer._digest(iet, configuration))

            # Create a YASK compiler solution for this Operator
            yc_soln = context.make_yc_solution(name)

            try:
                trees = offloadable.trees

                # Generate YASK grids and populate `yc_soln` with equations
                mapper = yaskizer(trees, yc_soln)
                local_grids = [i for i in mapper if i.is_Array]

                # Transform the IET
                funcall = make_sharedptr_funcall(namespace['code-soln-run'], ['time'],
                                                 namespace['code-soln-name'])
                funcall = Element(c.Statement(ccode(funcall)))
                mapper = {trees[0].root: funcall}
                mapper.update({i.root: mapper.get(i.root) for i in trees})  # Drop trees
                iet = Transformer(mapper).visit(iet)

                # Mark `funcall` as an external function call
                self.func_table[namespace['code-soln-run']] = MetaCall(None, False)

                # JIT-compile the newly-created YASK kernel
                self.yk_soln = context.make_yk_solution(name, yc_soln, local_grids)

                # Print some useful information about the newly constructed solution
                log("Solution '%s' contains %d grid(s) and %d equation(s)." %
                    (yc_soln.get_name(), yc_soln.get_num_grids(),
                     yc_soln.get_num_equations()))
            except NotImplementedError as e:
                self.yk_soln = YaskNullKernel()

                log("Unable to offload a candidate tree. Reason: [%s]" % str(e))

        # Some Iteration/Expression trees are not offloaded to YASK and may
        # require further processing to be executed in YASK, due to the differences
        # in storage layout employed by Devito and YASK
        iet = make_grid_accesses(iet)

        # Finally optimize all non-yaskized loops
        iet = super(Operator, self)._specialize_iet(iet, **kwargs)

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

    def _compile(self):
        """
        JIT-compile the C code generated by the Operator.

        It is ensured that JIT compilation will only be performed once per
        :class:`Operator`, reagardless of how many times this method is invoked.

        :returns: The file name of the JIT-compiled function.
        """
        if self._lib is None:
            if not isinstance(self.yk_soln, YaskNullKernel):
                self._compiler.libraries.append(self.yk_soln.soname)
            jit_compile(self._soname, str(self.ccode), self._compiler)


def find_offloadable_trees(iet):
    """
    Return the trees within ``iet`` that can be computed by YASK.

    A tree is "offloadable to YASK" if it is embedded in a time stepping loop
    *and* all of the grids accessed by the enclosed equations are homogeneous
    (i.e., same dimensions and data type).
    """
    Offloadable = namedtuple('Offlodable', 'trees grid dtype')
    Offloadable.__new__.__defaults__ = [], None, None

    reducer = ReducerMap()

    # Find offloadable candidates
    for tree in retrieve_iteration_tree(iet):
        # The outermost iteration must be over time and it must
        # nest at least one iteration
        if len(tree) <= 1:
            continue
        if not tree.root.dim.is_Time:
            continue
        grid_tree = tree[1:]
        if not all(i.is_Affine for i in grid_tree):
            # Non-affine array accesses unsupported by YASK
            continue
        bundles = [i for i in tree.inner.nodes if i.is_ExpressionBundle]
        if len(bundles) != 1:
            # Illegal nest
            continue
        bundle = bundles[0]
        # Found an offloadable candidate
        reducer.setdefault('grid_trees', []).append(grid_tree)
        # Track `grid` and `dtype`
        functions = flatten(i.functions for i in bundle.exprs)
        reducer.extend(('grid', i.grid) for i in functions if i.is_TimeFunction)
        reducer.extend(('dtype', i.dtype) for i in functions if i.is_TimeFunction)

    # `grid` and `dtype` must be unique
    try:
        grid = reducer.unique('grid')
        dtype = reducer.unique('dtype')
        trees = reducer['grid_trees']
    except (KeyError, ValueError):
        return Offloadable()

    # Do the trees iterate over a convex section of the grid?
    if not all(i.dim._defines | set(grid.dimensions) for i in flatten(trees)):
        return Offloadable()

    return Offloadable(trees, grid, dtype)

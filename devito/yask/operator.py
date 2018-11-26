import ctypes
from collections import OrderedDict
from pathlib import Path

import numpy as np

from devito.logger import yask as log
from devito.ir.equations import LoweredEq
from devito.ir.iet import (Iteration, Expression, ForeignExpression, Section,
                           MetaCall, FindNodes, Transformer, retrieve_iteration_tree)
from devito.ir.support import align_accesses
from devito.operator import OperatorRunnable
from devito.tools import ReducerMap, Signer, filter_ordered, flatten

from devito.yask import configuration
from devito.yask.data import DataScalar
from devito.yask.utils import make_grid_accesses, make_sharedptr_funcall, namespace
from devito.yask.wrappers import contexts
from devito.yask.transformer import yaskit
from devito.yask.types import YaskGridObject, YaskSolnObject

__all__ = ['Operator']


class Operator(OperatorRunnable):

    """
    A special :class:`OperatorCore` to JIT-compile and run operators through YASK.
    """

    _default_headers = OperatorRunnable._default_headers + ['#define restrict __restrict']
    _default_includes = OperatorRunnable._default_includes + ['yask_kernel_api.hpp']

    def __init__(self, expressions, **kwargs):
        super(Operator, self).__init__(expressions, **kwargs)
        # Each YASK Operator needs to have its own compiler (hence the copy()
        # below) because Operator-specific shared object are added to the
        # list of linked libraries
        self._compiler = configuration.yask['compiler'].copy()
        self._compiler.libraries.extend([i.soname for i in self.yk_solns.values()])

    def _specialize_exprs(self, expressions):
        # Align data accesses to the computational domain if not a yask.Function
        key = lambda i: i.is_TensorFunction and not i.from_YASK
        expressions = [align_accesses(e, key=key) for e in expressions]

        expressions = super(Operator, self)._specialize_exprs(expressions)

        # No matter whether offloading will occur or not, all YASK grids accept
        # negative indices when using the get/set_element_* methods (up to the
        # padding extent), so the OOB-relative data space should be adjusted
        return [LoweredEq(e,
                          dspace=e.dspace.zero([d for d in e.dimensions if d.is_Space]))
                for e in expressions]

    def _specialize_iet(self, iet, **kwargs):
        """
        Transform the Iteration/Expression tree to offload the computation of
        one or more loop nests onto YASK. This involves calling the YASK compiler
        to generate YASK code. Such YASK code is then called from within the
        transformed Iteration/Expression tree.
        """
        mapper = {}
        self.yk_solns = OrderedDict()
        for n, (section, trees) in enumerate(find_offloadable_trees(iet).items()):
            dimensions = tuple(filter_ordered(i.dim.root for i in flatten(trees)))
            context = contexts.fetch(dimensions, self._dtype)

            # A unique name for the 'real' compiler and kernel solutions
            name = namespace['jit-soln'](Signer._digest(configuration,
                                                        *[i.root for i in trees]))

            # Create a YASK compiler solution for this Operator
            yc_soln = context.make_yc_solution(name)

            try:
                # Generate YASK grids and populate `yc_soln` with equations
                gridmap = yaskit(trees, yc_soln)
                local_grids = [i for i in gridmap if i.is_Array]

                # Build the new IET nodes
                yk_soln_obj = YaskSolnObject(namespace['code-soln-name'](n))
                funcall = make_sharedptr_funcall(namespace['code-soln-run'],
                                                 ['time'], yk_soln_obj)
                funcall = ForeignExpression(funcall, self._dtype)
                mapper[trees[0].root] = funcall
                mapper.update({i.root: mapper.get(i.root) for i in trees})  # Drop trees

                # Mark `funcall` as an external function call
                self._func_table[namespace['code-soln-run']] = MetaCall(None, False)

                # JIT-compile the newly-created YASK kernel
                yk_soln = context.make_yk_solution(name, yc_soln, local_grids)
                self.yk_solns[(dimensions, yk_soln_obj)] = yk_soln

                # Print some useful information about the newly constructed solution
                log("Solution '%s' contains %d grid(s) and %d equation(s)." %
                    (yc_soln.get_name(), yc_soln.get_num_grids(),
                     yc_soln.get_num_equations()))
            except NotImplementedError as e:
                log("Unable to offload a candidate tree. Reason: [%s]" % str(e))
        iet = Transformer(mapper).visit(iet)

        if not self.yk_solns:
            log("No offloadable trees found")

        # Some Iteration/Expression trees are not offloaded to YASK and may
        # require further processing to be executed in YASK, due to the differences
        # in storage layout employed by Devito and YASK
        yk_grid_objs = {i.name: YaskGridObject(i.name) for i in self.input if i.from_YASK}
        yk_grid_objs.update({i: YaskGridObject(i) for i in self._local_grids})
        iet = make_grid_accesses(iet, yk_grid_objs)

        # Finally optimize all non-yaskized loops
        iet = super(Operator, self)._specialize_iet(iet, **kwargs)

        return iet

    def _build_parameters(self, iet):
        parameters = super(Operator, self)._build_parameters(iet)
        # Add parameters "disappeared" due to offloading
        parameters += tuple(i for i in self.input if i not in parameters)
        return parameters

    @property
    def _local_grids(self):
        ret = {}
        for i in self.yk_solns.values():
            ret.update(i.local_grids)
        return ret

    def arguments(self, **kwargs):
        args = {}
        # Add in solution pointers
        args.update({i.name: v.rawpointer for (_, i), v in self.yk_solns.items()})
        # Add in local grids pointers
        for k, v in self._local_grids.items():
            args[namespace['code-grid-name'](k)] = ctypes.cast(int(v),
                                                               namespace['type-grid'])
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

        for i in self.yk_solns.values():
            i.pre_apply(toshare)
        arg_values = [args[p.name] for p in self.parameters]
        self.cfunction(*arg_values)
        for i in self.yk_solns.values():
            i.post_apply()

        # Output summary of performance achieved
        return self._profile_output(args)

    def __getstate__(self):
        state = dict(super(Operator, self).__getstate__())
        # A YASK solution object needs to be recreated upon unpickling. Steps:
        # 1) upon pickling: serialise all files generated by this Operator via YASK
        # 2) upon unpickling: deserialise and explicitly recreate the YASK solution
        state['yk_solns'] = []
        for (dimensions, yk_soln_obj), yk_soln in self.yk_solns.items():
            path = Path(namespace['yask-lib'], 'lib%s.so' % yk_soln.soname)
            with open(path, 'rb') as f:
                libfile = f.read()
            path = Path(namespace['yask-pylib'], '%s.py' % yk_soln.name)
            with open(path, 'r') as f:
                pyfile = f.read().encode()
            path = Path(namespace['yask-pylib'], '_%s.so' % yk_soln.name)
            with open(path, 'rb') as f:
                pysofile = f.read()
            state['yk_solns'].append((dimensions, yk_soln_obj, yk_soln.name,
                                      yk_soln.soname, libfile, pyfile,
                                      pysofile, list(yk_soln.local_grids)))
        return state

    def __setstate__(self, state):
        yk_solns = state.pop('yk_solns')
        super(Operator, self).__setstate__(state)
        # Restore the YASK solutions (see __getstate__ for more info)
        self.yk_solns = OrderedDict()
        for (dimensions, yk_soln_obj, name, soname,
             libfile, pyfile, pysofile, local_grids) in yk_solns:
            path = Path(namespace['yask-lib'], 'lib%s.so' % soname)
            if not path.is_file():
                with open(path, 'wb') as f:
                    f.write(libfile)
            path = Path(namespace['yask-pylib'], '%s.py' % name)
            if not path.is_file():
                with open(path, 'w') as f:
                    f.write(pyfile.decode())
            path = Path(namespace['yask-pylib'], '_%s.so' % name)
            if not path.is_file():
                with open(path, 'wb') as f:
                    f.write(pysofile)
            # Finally reinstantiate the YASK solution -- no code generation or JIT
            # will happen at this point, as all necessary files have been restored
            context = contexts.fetch(dimensions, self._dtype)
            local_grids = [i for i in self.parameters if i.name in local_grids]
            yk_soln = context.make_yk_solution(name, None, local_grids)
            self.yk_solns[(dimensions, yk_soln_obj)] = yk_soln


def find_offloadable_trees(iet):
    """
    Return the trees within ``iet`` that can be computed by YASK.

    A tree is "offloadable to YASK" if it is embedded in a time stepping loop
    *and* all of the grids accessed by the enclosed equations are homogeneous
    (i.e., same dimensions and data type).
    """
    offloadable = OrderedDict()
    roots = [i for i in FindNodes(Iteration).visit(iet) if i.dim.is_Time]
    for root in roots:
        sections = FindNodes(Section).visit(root)
        for section in sections:
            for tree in retrieve_iteration_tree(section):
                if not all(i.is_Affine for i in tree):
                    # Non-affine array accesses unsupported by YASK
                    break
                exprs = [i.expr for i in FindNodes(Expression).visit(tree.root)]
                grid = ReducerMap([('', i.grid) for i in exprs if i.grid]).unique('')
                writeto_dimensions = tuple(i.dim.root for i in tree)
                if grid.dimensions == writeto_dimensions:
                    offloadable.setdefault(section, []).append(tree)
                else:
                    break
    return offloadable

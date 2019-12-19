import ctypes
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np

from devito.exceptions import InvalidOperator
from devito.logger import yask as log
from devito.ir.clusters import Toposort
from devito.ir.equations import LoweredEq
from devito.ir.iet import (Expression, FindNodes, FindSymbols, Transformer,
                           derive_parameters, find_affine_trees)
from devito.ir.support import align_accesses
from devito.operator import Operator
from devito.passes.clusters import Lift, fuse, scalarize, eliminate_arrays, rewrite
from devito.passes.iet import (DataManager, Ompizer, avoid_denormals, loop_wrapping,
                               iet_pass)
from devito.tools import Signer, as_tuple, filter_ordered, flatten, generator

from devito.yask import configuration
from devito.yask.data import DataScalar
from devito.yask.utils import (Offloaded, make_var_accesses, make_sharedptr_funcall,
                               namespace)
from devito.yask.wrappers import contexts
from devito.yask.transformer import yaskit
from devito.yask.types import YASKVarObject, YASKSolnObject


__all__ = ['YASKNoopOperator', 'YASKOperator', 'YASKCustomOperator']


@iet_pass
def make_yask_kernels(iet, **kwargs):
    yk_solns = kwargs.pop('yk_solns')

    mapper = {}
    for n, (section, trees) in enumerate(find_affine_trees(iet).items()):
        dimensions = tuple(filter_ordered(i.dim.root for i in flatten(trees)))

        # Retrieve the section dtype
        exprs = FindNodes(Expression).visit(section)
        dtypes = {e.dtype for e in exprs}
        if len(dtypes) != 1:
            log("Unable to offload in presence of mixed-precision arithmetic")
            continue
        dtype = dtypes.pop()

        context = contexts.fetch(dimensions, dtype)

        # A unique name for the 'real' compiler and kernel solutions
        name = namespace['jit-soln'](Signer._digest(configuration,
                                                    *[i.root for i in trees]))

        # Create a YASK compiler solution for this Operator
        yc_soln = context.make_yc_solution(name)

        try:
            # Generate YASK vars and populate `yc_soln` with equations
            local_vars = yaskit(trees, yc_soln)

            # Build the new IET nodes
            yk_soln_obj = YASKSolnObject(namespace['code-soln-name'](n))
            funcall = make_sharedptr_funcall(namespace['code-soln-run'],
                                             ['time'], yk_soln_obj)
            funcall = Offloaded(funcall, dtype)
            mapper[trees[0].root] = funcall
            mapper.update({i.root: mapper.get(i.root) for i in trees})  # Drop trees

            # JIT-compile the newly-created YASK kernel
            yk_soln = context.make_yk_solution(name, yc_soln, local_vars)
            yk_solns[(dimensions, yk_soln_obj)] = yk_soln

            # Print some useful information about the newly constructed solution
            log("Solution '%s' contains %d var(s) and %d equation(s)." %
                (yc_soln.get_name(), yc_soln.get_num_vars(),
                 yc_soln.get_num_equations()))
        except NotImplementedError as e:
            log("Unable to offload a candidate tree. Reason: [%s]" % str(e))
    iet = Transformer(mapper).visit(iet)

    if not yk_solns:
        log("No offloadable trees found")

    # Some Iteration/Expression trees are not offloaded to YASK and may
    # require further processing to be executed through YASK, due to the
    # different storage layout
    yk_var_objs = {i.name: YASKVarObject(i.name)
                   for i in FindSymbols().visit(iet) if i.from_YASK}
    yk_var_objs.update({i: YASKVarObject(i) for i in get_local_vars(yk_solns)})
    iet = make_var_accesses(iet, yk_var_objs)

    # The signature needs to be updated
    # TODO: this could be done automagically through the iet pass engine, but
    # currently it only supports *appending* to the parameters list. While here
    # we actually need to change it as some parameters may disappear (x_m, x_M, ...)
    parameters = derive_parameters(iet, True)
    iet = iet._rebuild(parameters=parameters)

    return iet, {}


class YASKOmpizer(Ompizer):

    def __init__(self, key=None):
        if key is None:
            def key(i):
                # If it's not parallel, nothing to do
                if not i.is_ParallelRelaxed or i.is_Vectorized:
                    return False
                # If some of the inner computation has been offloaded to YASK,
                # avoid introducing an outer level of parallelism
                if FindNodes(Offloaded).visit(i):
                    return False
                return True
        super(YASKOmpizer, self).__init__(key=key)


class YASKOperator(Operator):

    """
    A special Operator generating and executing YASK code.
    """

    _default_headers = Operator._default_headers + ['#define restrict __restrict']
    _default_includes = Operator._default_includes + ['yask_kernel_api.hpp']

    @classmethod
    def _build(cls, expressions, **kwargs):
        yk_solns = OrderedDict()
        op = super(YASKOperator, cls)._build(expressions, yk_solns=yk_solns, **kwargs)

        # Produced by `_specialize_iet`
        op.yk_solns = yk_solns

        # Each YASK Operator needs to have its own compiler (hence the copy()
        # below) because Operator-specific shared object are added to the
        # list of linked libraries
        op._compiler = configuration.yask['compiler'].copy()
        op._compiler.libraries.extend([i.soname for i in op.yk_solns.values()])

        return op

    @classmethod
    def _specialize_exprs(cls, expressions):
        # Align data accesses to the computational domain if not a yask.Function
        key = lambda i: i.is_DiscreteFunction and not i.from_YASK
        expressions = [align_accesses(e, key=key) for e in expressions]

        expressions = super(YASKOperator, cls)._specialize_exprs(expressions)

        # No matter whether offloading will occur or not, all YASK vars accept
        # negative indices when using the get/set_element_* methods (up to the
        # padding extent), so the OOB-relative data space should be adjusted
        return [LoweredEq(e,
                          dspace=e.dspace.zero([d for d in e.dimensions if d.is_Space]))
                for e in expressions]

    @classmethod
    def _specialize_clusters(cls, clusters, **kwargs):
        # TODO: this is currently identical to CPU64NoopOperator._specialize_clusters,
        # but it will have to change

        # To create temporaries
        counter = generator()
        template = lambda: "r%d" % counter()

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = Toposort().process(clusters)
        clusters = fuse(clusters)

        # Flop reduction via the DSE
        clusters = rewrite(clusters, template, **kwargs)

        # Lifting
        clusters = Lift().process(clusters)

        # Lifting may create fusion opportunities, which in turn may enable
        # further optimizations
        clusters = fuse(clusters)
        clusters = eliminate_arrays(clusters, template)
        clusters = scalarize(clusters, template)

        return clusters

    @classmethod
    def _specialize_iet(cls, graph, **kwargs):
        """
        Transform the Iteration/Expression tree to offload the computation of
        one or more loop nests onto YASK. This involves calling the YASK compiler
        to generate YASK code. Such YASK code is then called from within the
        transformed Iteration/Expression tree.
        """
        options = kwargs['options']
        yk_solns = kwargs['yk_solns']

        # Flush denormal numbers
        avoid_denormals(graph)

        # Create YASK kernels
        make_yask_kernels(graph, yk_solns=yk_solns)

        # Shared-memory and SIMD-level parallelism
        if options['openmp']:
            YASKOmpizer().make_parallel(graph)

        # Misc optimizations
        loop_wrapping(graph)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph

    def arguments(self, **kwargs):
        args = {}
        # Add in solution pointers
        args.update({i.name: v.rawpointer for (_, i), v in self.yk_solns.items()})
        # Add in local vars pointers
        for k, v in get_local_vars(self.yk_solns).items():
            args[namespace['code-var-name'](k)] = ctypes.cast(int(v),
                                                              namespace['type-var'])
        return super(YASKOperator, self).arguments(backend=args, **kwargs)

    def apply(self, **kwargs):
        # Build the arguments list to invoke the kernel function
        args = self.arguments(**kwargs)

        # Map default Functions to runtime Functions; will be used for "var sharing"
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
        cfunction = self.cfunction
        with self._profiler.timer_on('apply', comm=args.comm):
            cfunction(*arg_values)

        for i in self.yk_solns.values():
            i.post_apply()

        # Output summary of performance achieved
        return self._emit_apply_profiling(args)

    def __getstate__(self):
        state = dict(super(YASKOperator, self).__getstate__())
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
                                      pysofile, list(yk_soln.local_vars)))
        return state

    def __setstate__(self, state):
        yk_solns = state.pop('yk_solns')
        super(YASKOperator, self).__setstate__(state)
        # Restore the YASK solutions (see __getstate__ for more info)
        self.yk_solns = OrderedDict()
        for (dimensions, yk_soln_obj, name, soname,
             libfile, pyfile, pysofile, local_vars) in yk_solns:
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
            local_vars = [i for i in self.parameters if i.name in local_vars]
            yk_soln = context.make_yk_solution(name, None, local_vars)
            self.yk_solns[(dimensions, yk_soln_obj)] = yk_soln


class YASKNoopOperator(YASKOperator):

    @classmethod
    def _specialize_iet(cls, graph, **kwargs):
        yk_solns = kwargs['yk_solns']

        # Create YASK kernels
        make_yask_kernels(graph, yk_solns=yk_solns)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


class YASKCustomOperator(YASKOperator):

    @classmethod
    def _make_passes_mapper(cls, **kwargs):
        ompizer = YASKOmpizer()

        return {
            'denormals': partial(avoid_denormals),
            'wrapping': partial(loop_wrapping),
            'openmp': partial(ompizer.make_parallel),
        }

    @classmethod
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        passes = as_tuple(kwargs['mode'])

        # Create YASK kernels
        make_yask_kernels(graph, **kwargs)

        # Fetch passes to be called
        passes_mapper = cls._make_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                passes_mapper[i](graph)
            except KeyError:
                raise InvalidOperator("Unknown passes `%s`" % str(passes))

        # Force-call `openmp` if requested via global option
        if 'openmp' not in passes and options['openmp']:
            passes_mapper['openmp'](graph)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


# Utility functions

def get_local_vars(yk_solns):
    ret = {}
    for i in yk_solns.values():
        ret.update(i.local_vars)
    return ret

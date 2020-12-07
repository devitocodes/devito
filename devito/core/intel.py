from functools import partial

from devito.core.cpu import CPU64Operator, CPU64OpenMPOperator
from devito.exceptions import InvalidOperator
from devito.passes.equations import collect_derivatives
from devito.passes.clusters import (Blocking, Lift, cire, cse, eliminate_arrays,
                                    extract_increments, factorize, fuse, optimize_pows)
from devito.passes.iet import (DataManager, Ompizer, avoid_denormals, mpiize,
                               optimize_halospots, hoist_prodders, relax_incr_dimensions)
from devito.tools import as_tuple, timed_pass

__all__ = ['Intel64Operator', 'Intel64OpenMPOperator', 'Intel64FSGOperator',
           'Intel64FSGOpenMPOperator']


Intel64Operator = CPU64Operator
Intel64OpenMPOperator = CPU64OpenMPOperator


class Intel64FSGOperator(Intel64Operator):

    """
    Operator with performance optimizations tailored "For Small Grids" (FSG).
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        kwargs = super(Intel64FSGOperator, cls)._normalize_kwargs(**kwargs)

        if kwargs['options']['min-storage']:
            raise InvalidOperator('You should not use `min-storage` with `advanced-fsg '
                                  ' as they work in opposite directions')

        return kwargs

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = fuse(clusters, toposort=True)

        # Hoist and optimize Dimension-invariant sub-expressions
        clusters = cire(clusters, 'invariants', sregistry, options, platform)
        clusters = Lift().process(clusters)

        # Reduce flops (potential arithmetic alterations)
        clusters = extract_increments(clusters, sregistry)
        clusters = cire(clusters, 'sops', sregistry, options, platform)
        clusters = factorize(clusters)
        clusters = optimize_pows(clusters)

        # The previous passes may have created fusion opportunities, which in
        # turn may enable further optimizations
        clusters = fuse(clusters)
        clusters = eliminate_arrays(clusters)

        # Reduce flops (no arithmetic alterations)
        clusters = cse(clusters, sregistry)

        # Blocking to improve data locality
        clusters = Blocking(options).process(clusters)

        return clusters


class Intel64FSGOpenMPOperator(Intel64FSGOperator, CPU64OpenMPOperator):
    _specialize_iet = CPU64OpenMPOperator._specialize_iet


PowerOperator = CPU64Operator
PowerOpenMPOperator = CPU64OpenMPOperator

ArmOperator = CPU64Operator
ArmOpenMPOperator = CPU64OpenMPOperator


class CustomOperator(CPU64Operator):

    _known_passes = ('blocking', 'denormals', 'optcomms', 'openmp', 'mpi',
                     'simd', 'prodders', 'topofuse', 'fuse', 'factorize',
                     'cire-sops', 'cse', 'lift', 'opt-pows', 'collect-derivs')

    @classmethod
    def _make_exprs_passes_mapper(cls, **kwargs):
        return {
            'collect-derivs': collect_derivatives,
        }

    @classmethod
    def _make_clusters_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        return {
            'blocking': Blocking(options).process,
            'factorize': factorize,
            'fuse': fuse,
            'lift': lambda i: Lift().process(cire(i, 'invariants', sregistry,
                                                  options, platform)),
            'cire-sops': lambda i: cire(i, 'sops', sregistry, options, platform),
            'cse': lambda i: cse(i, sregistry),
            'opt-pows': optimize_pows,
            'topofuse': lambda i: fuse(i, toposort=True)
        }

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        ompizer = Ompizer(sregistry, options)

        return {
            'denormals': avoid_denormals,
            'optcomms': optimize_halospots,
            'blocking': partial(relax_incr_dimensions, sregistry=sregistry),
            'openmp': ompizer.make_parallel,
            'mpi': partial(mpiize, mode=options['mpi']),
            'simd': partial(ompizer.make_simd, simd_reg_size=platform.simd_reg_size),
            'prodders': hoist_prodders
        }

    @classmethod
    def _build(cls, expressions, **kwargs):
        # Sanity check
        passes = as_tuple(kwargs['mode'])
        if any(i not in cls._known_passes for i in passes):
            raise InvalidOperator("Unknown passes `%s`" % str(passes))

        return super(CustomOperator, cls)._build(expressions, **kwargs)

    @classmethod
    @timed_pass(name='specializing.Expressions')
    def _specialize_exprs(cls, expressions, **kwargs):
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_exprs_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                expressions = passes_mapper[i](expressions)
            except KeyError:
                pass

        return expressions

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_clusters_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                clusters = passes_mapper[i](clusters)
            except KeyError:
                pass

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_iet_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                passes_mapper[i](graph)
            except KeyError:
                pass

        # Force-call `mpi` if requested via global option
        if 'mpi' not in passes and options['mpi']:
            passes_mapper['mpi'](graph)

        # Force-call `openmp` if requested via global option
        if 'openmp' not in passes and options['openmp']:
            passes_mapper['openmp'](graph)

        # Symbol definitions
        data_manager = DataManager(sregistry)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph

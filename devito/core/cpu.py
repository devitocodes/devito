from functools import partial

from devito.core.operator import OperatorCore
from devito.exceptions import InvalidOperator
from devito.passes.equations import collect_derivatives
from devito.passes.clusters import (Blocking, Lift, cire, cse, eliminate_arrays,
                                    extract_increments, factorize, fuse, optimize_pows)
from devito.passes.iet import (DataManager, Ompizer, avoid_denormals, mpiize,
                               optimize_halospots, hoist_prodders, relax_incr_dimensions)
from devito.tools import as_tuple, timed_pass

__all__ = ['CPU64NoopOperator', 'CPU64Operator', 'CPU64OpenMPOperator',
           'Intel64Operator', 'Intel64OpenMPOperator', 'Intel64FSGOperator',
           'Intel64FSGOpenMPOperator',
           'PowerOperator', 'PowerOpenMPOperator',
           'ArmOperator', 'ArmOpenMPOperator',
           'CustomOperator']


class CPU64NoopOperator(OperatorCore):

    BLOCK_LEVELS = 1
    """
    Loop blocking depth. So, 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    CIRE_REPEATS_INV = 1
    """
    Number of CIRE passes to detect and optimize away Dimension-invariant expressions.
    """

    CIRE_REPEATS_SOPS = 7
    """
    Number of CIRE passes to detect and optimize away redundant sum-of-products.
    """

    CIRE_MINCOST_INV = 50
    """
    Minimum operation count of a Dimension-invariant aliasing expression to be
    optimized away. Dimension-invariant aliases are lifted outside of one or more
    invariant loop(s), so they require tensor temporaries that can be potentially
    very large (e.g., the whole domain in the case of time-invariant aliases).
    """

    CIRE_MINCOST_SOPS = 10
    """
    Minimum operation count of a sum-of-product aliasing expression to be optimized away.
    """

    PAR_COLLAPSE_NCORES = 4
    """
    Use a collapse clause if the number of available physical cores is greater
    than this threshold.
    """

    PAR_COLLAPSE_WORK = 100
    """
    Use a collapse clause if the trip count of the collapsable loops is statically
    known to exceed this threshold.
    """

    PAR_CHUNK_NONAFFINE = 3
    """
    Coefficient to adjust the chunk size in non-affine parallel loops.
    """

    PAR_DYNAMIC_WORK = 10
    """
    Use dynamic scheduling if the operation count per iteration exceeds this
    threshold. Otherwise, use static scheduling.
    """

    PAR_NESTED = 2
    """
    Use nested parallelism if the number of hyperthreads per core is greater
    than this threshold.
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        o = {}
        oo = kwargs['options']

        # Execution modes
        o['openmp'] = oo.pop('openmp')
        o['mpi'] = oo.pop('mpi')

        # Blocking
        o['blockinner'] = oo.pop('blockinner', False)
        o['blocklevels'] = oo.pop('blocklevels', cls.BLOCK_LEVELS)

        # CIRE
        o['min-storage'] = oo.pop('min-storage', False)
        o['cire-rotate'] = oo.pop('cire-rotate', False)
        o['cire-onstack'] = oo.pop('cire-onstack', False)
        o['cire-maxpar'] = oo.pop('cire-maxpar', False)
        o['cire-maxalias'] = oo.pop('cire-maxalias', False)
        o['cire-repeats'] = {
            'invariants': oo.pop('cire-repeats-inv', cls.CIRE_REPEATS_INV),
            'sops': oo.pop('cire-repeats-sops', cls.CIRE_REPEATS_SOPS)
        }
        o['cire-mincost'] = {
            'invariants': oo.pop('cire-mincost-inv', cls.CIRE_MINCOST_INV),
            'sops': oo.pop('cire-mincost-sops', cls.CIRE_MINCOST_SOPS)
        }

        # Shared-memory parallelism
        o['par-collapse-ncores'] = oo.pop('par-collapse-ncores', cls.PAR_COLLAPSE_NCORES)
        o['par-collapse-work'] = oo.pop('par-collapse-work', cls.PAR_COLLAPSE_WORK)
        o['par-chunk-nonaffine'] = oo.pop('par-chunk-nonaffine', cls.PAR_CHUNK_NONAFFINE)
        o['par-dynamic-work'] = oo.pop('par-dynamic-work', cls.PAR_DYNAMIC_WORK)
        o['par-nested'] = oo.pop('par-nested', cls.PAR_NESTED)

        o['gpu-direct'] = oo.pop('gpu-direct', False)

        if oo:
            raise InvalidOperator("Unrecognized optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Shared-memory parallelism
        if options['openmp']:
            ompizer = Ompizer(sregistry, options)
            ompizer.make_parallel(graph)

        # Symbol definitions
        data_manager = DataManager(sregistry)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


class CPU64Operator(CPU64NoopOperator):

    @classmethod
    @timed_pass(name='specializing.Expressions')
    def _specialize_exprs(cls, expressions, **kwargs):
        expressions = collect_derivatives(expressions)

        return expressions

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

        # Blocking to improve data locality
        clusters = Blocking(options).process(clusters)

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

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Flush denormal numbers
        avoid_denormals(graph)

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Lower IncrDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph, sregistry=sregistry)

        # SIMD-level parallelism
        ompizer = Ompizer(sregistry, options)
        ompizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DataManager(sregistry)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


class CPU64OpenMPOperator(CPU64Operator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Flush denormal numbers
        avoid_denormals(graph)

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Lower IncrDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph, sregistry=sregistry)

        # SIMD-level parallelism
        ompizer = Ompizer(sregistry, options)
        ompizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)

        # Shared-memory parallelism
        ompizer.make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DataManager(sregistry)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


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


class ArmOperator(CPU64Operator):
    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Lower IncrDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph, sregistry=sregistry)

        # SIMD-level parallelism
        ompizer = Ompizer(sregistry, options)
        ompizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DataManager(sregistry)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


class ArmOpenMPOperator(CPU64OpenMPOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Lower IncrDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph, sregistry=sregistry)

        # SIMD-level parallelism
        ompizer = Ompizer(sregistry, options)
        ompizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)

        # Shared-memory parallelism
        ompizer.make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DataManager(sregistry)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


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

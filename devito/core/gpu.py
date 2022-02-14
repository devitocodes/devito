from functools import partial

import numpy as np

from devito.core.operator import CoreOperator, CustomOperator, ParTile
from devito.exceptions import InvalidOperator
from devito.passes.equations import collect_derivatives
from devito.passes.clusters import (Lift, Streaming, Tasker, blocking, buffering,
                                    cire, cse, extract_increments, factorize,
                                    fission, fuse, optimize_pows, optimize_msds)
from devito.passes.iet import (DeviceOmpTarget, DeviceAccTarget, mpiize, hoist_prodders,
                               is_on_device, linearize, relax_incr_dimensions)
from devito.tools import as_tuple, timed_pass

__all__ = ['DeviceNoopOperator', 'DeviceAdvOperator', 'DeviceCustomOperator',
           'DeviceNoopOmpOperator', 'DeviceAdvOmpOperator', 'DeviceFsgOmpOperator',
           'DeviceCustomOmpOperator', 'DeviceNoopAccOperator', 'DeviceAdvAccOperator',
           'DeviceFsgAccOperator', 'DeviceCustomAccOperator']


class DeviceOperatorMixin(object):

    BLOCK_LEVELS = 0
    """
    Loop blocking depth. So, 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    BLOCK_EAGER = True
    """
    Apply loop blocking as early as possible, and in particular prior to CIRE.
    """

    BLOCK_RELAX = False
    """
    If set to True, bypass the compiler heuristics that prevent loop blocking in
    situations where the performance impact might be detrimental.
    """

    CIRE_MINGAIN = 10
    """
    Minimum operation count reduction for a redundant expression to be optimized
    away. Higher (lower) values make a redundant expression less (more) likely to
    be optimized away.
    """

    CIRE_SCHEDULE = 'automatic'
    """
    Strategy used to schedule derivatives across loops. This impacts the operational
    intensity of the generated kernel.
    """

    PAR_CHUNK_NONAFFINE = 3
    """
    Coefficient to adjust the chunk size in non-affine parallel loops.
    """

    GPU_FIT = 'all-fallback'
    """
    Assuming all functions fit into the gpu memory.
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        o = {}
        oo = kwargs['options']

        # Execution modes
        o['mpi'] = oo.pop('mpi')
        o['parallel'] = True

        # Buffering
        o['buf-async-degree'] = oo.pop('buf-async-degree', None)

        # Fusion
        o['fuse-tasks'] = oo.pop('fuse-tasks', False)

        # Blocking
        o['blockinner'] = oo.pop('blockinner', True)
        o['blocklevels'] = oo.pop('blocklevels', cls.BLOCK_LEVELS)
        o['blockeager'] = oo.pop('blockeager', cls.BLOCK_EAGER)
        o['blocklazy'] = oo.pop('blocklazy', not o['blockeager'])
        o['blockrelax'] = oo.pop('blockrelax', cls.BLOCK_RELAX)
        o['skewing'] = oo.pop('skewing', False)

        # CIRE
        o['min-storage'] = False
        o['cire-rotate'] = False
        o['cire-maxpar'] = oo.pop('cire-maxpar', True)
        o['cire-ftemps'] = oo.pop('cire-ftemps', False)
        o['cire-mingain'] = oo.pop('cire-mingain', cls.CIRE_MINGAIN)
        o['cire-schedule'] = oo.pop('cire-schedule', cls.CIRE_SCHEDULE)

        # GPU parallelism
        o['par-tile'] = ParTile(oo.pop('par-tile', False), default=(32, 4))
        o['par-collapse-ncores'] = 1  # Always collapse (meaningful if `par-tile=False`)
        o['par-collapse-work'] = 1  # Always collapse (meaningful if `par-tile=False`)
        o['par-chunk-nonaffine'] = oo.pop('par-chunk-nonaffine', cls.PAR_CHUNK_NONAFFINE)
        o['par-dynamic-work'] = np.inf  # Always use static scheduling
        o['par-nested'] = np.inf  # Never use nested parallelism
        o['par-disabled'] = oo.pop('par-disabled', True)  # No host parallelism by default
        o['gpu-fit'] = as_tuple(oo.pop('gpu-fit', cls._normalize_gpu_fit(**kwargs)))

        # Misc
        o['optcomms'] = oo.pop('optcomms', True)
        o['linearize'] = oo.pop('linearize', False)

        if oo:
            raise InvalidOperator("Unsupported optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs

    @classmethod
    def _normalize_gpu_fit(cls, **kwargs):
        if any(i in kwargs['mode'] for i in ['tasking', 'streaming']):
            return None
        else:
            return cls.GPU_FIT

# Mode level


class DeviceNoopOperator(DeviceOperatorMixin, CoreOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        mpiize(graph, sregistry=sregistry, options=options)

        # GPU parallelism
        parizer = cls._Target.Parizer(sregistry, options, platform)
        parizer.make_parallel(graph)
        parizer.initialize(graph)

        # Symbol definitions
        cls._Target.DataManager(sregistry, options).process(graph)

        return graph


class DeviceAdvOperator(DeviceOperatorMixin, CoreOperator):

    @classmethod
    @timed_pass(name='specializing.DSL')
    def _specialize_dsl(cls, expressions, **kwargs):
        expressions = collect_derivatives(expressions)

        return expressions

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Optimize MultiSubDomains
        clusters = optimize_msds(clusters)

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = fuse(clusters, toposort=True, options=options)

        # Fission to increase parallelism
        clusters = fission(clusters)

        # Hoist and optimize Dimension-invariant sub-expressions
        clusters = cire(clusters, 'invariants', sregistry, options, platform)
        clusters = Lift().process(clusters)

        # Blocking to define thread blocks
        if options['blockeager']:
            clusters = blocking(clusters, sregistry, options)

        # Reduce flops
        clusters = extract_increments(clusters, sregistry)
        clusters = cire(clusters, 'sops', sregistry, options, platform)
        clusters = factorize(clusters)
        clusters = optimize_pows(clusters)

        # The previous passes may have created fusion opportunities
        clusters = fuse(clusters)

        # Reduce flops
        clusters = cse(clusters, sregistry)

        # Blocking to define thread blocks
        if options['blocklazy']:
            clusters = blocking(clusters, sregistry, options)

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        mpiize(graph, sregistry=sregistry, options=options)

        # Loop tiling
        relax_incr_dimensions(graph)

        # GPU parallelism
        parizer = cls._Target.Parizer(sregistry, options, platform)
        parizer.make_parallel(graph)
        parizer.initialize(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._Target.DataManager(sregistry, options).process(graph)

        # Linearize n-dimensional Indexeds
        linearize(graph, mode=options['linearize'], sregistry=sregistry)

        return graph


class DeviceFsgOperator(DeviceAdvOperator):

    """
    Operator with performance optimizations tailored "For small grids" ("Fsg").
    """

    # Note: currently mimics DeviceAdvOperator. Will see if this will change
    # in the future
    pass


class DeviceCustomOperator(DeviceOperatorMixin, CustomOperator):

    @classmethod
    def _make_dsl_passes_mapper(cls, **kwargs):
        return {
            'collect-derivs': collect_derivatives,
        }

    @classmethod
    def _make_clusters_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Callbacks used by `Tasking` and `Streaming`
        runs_on_host, reads_if_on_host = make_callbacks(options)

        # Callback used by `buffering`
        def callback(f):
            if not is_on_device(f, options['gpu-fit']):
                return [f.time_dim]
            else:
                return None

        return {
            'buffering': lambda i: buffering(i, callback, sregistry, options),
            'blocking': lambda i: blocking(i, sregistry, options),
            'tasking': Tasker(runs_on_host).process,
            'streaming': Streaming(reads_if_on_host).process,
            'factorize': factorize,
            'fission': fission,
            'fuse': lambda i: fuse(i, options=options),
            'lift': lambda i: Lift().process(cire(i, 'invariants', sregistry,
                                                  options, platform)),
            'cire-sops': lambda i: cire(i, 'sops', sregistry, options, platform),
            'cse': lambda i: cse(i, sregistry),
            'opt-pows': optimize_pows,
            'topofuse': lambda i: fuse(i, toposort=True, options=options)
        }

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        parizer = cls._Target.Parizer(sregistry, options, platform)
        orchestrator = cls._Target.Orchestrator(sregistry)

        return {
            'parallel': parizer.make_parallel,
            'orchestrate': partial(orchestrator.process),
            'mpi': partial(mpiize, sregistry=sregistry, options=options),
            'linearize': partial(linearize, mode=options['linearize'],
                                 sregistry=sregistry),
            'prodders': partial(hoist_prodders),
            'init': parizer.initialize
        }

    _known_passes = (
        # DSL
        'collect-derivs',
        # Expressions
        'buffering',
        # Clusters
        'blocking', 'tasking', 'streaming', 'factorize', 'fission', 'fuse', 'lift',
        'cire-sops', 'cse', 'opt-pows', 'topofuse',
        # IET
        'orchestrate', 'parallel', 'mpi', 'linearize', 'prodders'
    )
    _known_passes_disabled = ('denormals', 'simd')
    assert not (set(_known_passes) & set(_known_passes_disabled))


# Language level

# OpenMP

class DeviceOmpOperatorMixin(object):

    _Target = DeviceOmpTarget

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        oo = kwargs['options']

        oo.pop('openmp', None)  # It may or may not have been provided
        kwargs = super()._normalize_kwargs(**kwargs)
        oo['openmp'] = True

        return kwargs


class DeviceNoopOmpOperator(DeviceOmpOperatorMixin, DeviceNoopOperator):
    pass


class DeviceAdvOmpOperator(DeviceOmpOperatorMixin, DeviceAdvOperator):
    pass


class DeviceFsgOmpOperator(DeviceOmpOperatorMixin, DeviceFsgOperator):
    pass


class DeviceCustomOmpOperator(DeviceOmpOperatorMixin, DeviceCustomOperator):

    _known_passes = DeviceCustomOperator._known_passes + ('openmp',)
    assert not (set(_known_passes) & set(DeviceCustomOperator._known_passes_disabled))

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        mapper = super()._make_iet_passes_mapper(**kwargs)
        mapper['openmp'] = mapper['parallel']
        return mapper


# OpenACC

class DeviceAccOperatorMixin(object):

    _Target = DeviceAccTarget

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        oo = kwargs['options']
        oo.pop('openmp', None)

        kwargs = super()._normalize_kwargs(**kwargs)
        oo['openacc'] = True

        return kwargs


class DeviceNoopAccOperator(DeviceAccOperatorMixin, DeviceNoopOperator):
    pass


class DeviceAdvAccOperator(DeviceAccOperatorMixin, DeviceAdvOperator):
    pass


class DeviceFsgAccOperator(DeviceAccOperatorMixin, DeviceFsgOperator):
    pass


class DeviceCustomAccOperator(DeviceAccOperatorMixin, DeviceCustomOperator):

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        mapper = super()._make_iet_passes_mapper(**kwargs)
        mapper['openacc'] = mapper['parallel']
        return mapper

    _known_passes = DeviceCustomOperator._known_passes + ('openacc',)
    assert not (set(_known_passes) & set(DeviceCustomOperator._known_passes_disabled))


# Utils


def make_callbacks(options):
    """
    Options-dependent callbacks used by various compiler passes.
    """

    def is_on_host(f):
        return not is_on_device(f, options['gpu-fit'])

    def runs_on_host(c):
        # The only situation in which a Cluster doesn't get offloaded to
        # the device is when it writes to a host Function
        return any(is_on_host(f) for f in c.scope.writes)

    def reads_if_on_host(c):
        if not runs_on_host(c):
            return [f for f in c.scope.reads if is_on_host(f)]
        else:
            return []

    return runs_on_host, reads_if_on_host

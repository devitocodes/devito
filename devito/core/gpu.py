from functools import partial

import numpy as np

from devito.core.operator import CoreOperator, CustomOperator
from devito.exceptions import InvalidOperator
from devito.passes.equations import collect_derivatives, buffering
from devito.passes.clusters import (Blocking, Lift, Streaming, Tasker, cire, cse,
                                    eliminate_arrays, extract_increments, factorize,
                                    fuse, optimize_pows)
from devito.passes.iet import (DeviceOmpTarget, DeviceAccTarget, optimize_halospots,
                               mpiize, hoist_prodders, is_on_device)
from devito.tools import as_tuple, timed_pass

__all__ = ['DeviceNoopOperator', 'DeviceAdvOperator', 'DeviceCustomOperator',
           'DeviceNoopOmpOperator', 'DeviceAdvOmpOperator', 'DeviceCustomOmpOperator',
           'DeviceNoopAccOperator', 'DeviceAdvAccOperator', 'DeviceCustomAccOperator']


class DeviceOperatorMixin(object):

    BLOCK_LEVELS = 1
    """
    Loop blocking depth. So, 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    CIRE_REPEATS_INV = 2
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

    PAR_CHUNK_NONAFFINE = 3
    """
    Coefficient to adjust the chunk size in non-affine parallel loops.
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

        # Blocking
        o['blockinner'] = oo.pop('blockinner', True)
        o['blocklevels'] = oo.pop('blocklevels', cls.BLOCK_LEVELS)

        # CIRE
        o['min-storage'] = False
        o['cire-rotate'] = False
        o['cire-onstack'] = False
        o['cire-maxpar'] = oo.pop('cire-maxpar', True)
        o['cire-maxalias'] = oo.pop('cire-maxalias', False)
        o['cire-repeats'] = {
            'invariants': oo.pop('cire-repeats-inv', cls.CIRE_REPEATS_INV),
            'sops': oo.pop('cire-repeats-sops', cls.CIRE_REPEATS_SOPS)
        }
        o['cire-mincost'] = {
            'invariants': oo.pop('cire-mincost-inv', cls.CIRE_MINCOST_INV),
            'sops': oo.pop('cire-mincost-sops', cls.CIRE_MINCOST_SOPS)
        }

        # GPU parallelism
        o['par-collapse-ncores'] = 1  # Always use a collapse clause
        o['par-collapse-work'] = 1  # Always use a collapse clause
        o['par-chunk-nonaffine'] = oo.pop('par-chunk-nonaffine', cls.PAR_CHUNK_NONAFFINE)
        o['par-dynamic-work'] = np.inf  # Always use static scheduling
        o['par-nested'] = np.inf  # Never use nested parallelism
        o['par-disabled'] = oo.pop('par-disabled', True)  # No host parallelism by default
        o['gpu-direct'] = oo.pop('gpu-direct', True)
        o['gpu-fit'] = as_tuple(oo.pop('gpu-fit', None))

        if oo:
            raise InvalidOperator("Unsupported optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs


# Mode level


class DeviceNoopOperator(DeviceOperatorMixin, CoreOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # GPU parallelism
        parizer = cls._Target.Parizer(sregistry, options, platform)
        parizer.make_parallel(graph)

        # Symbol definitions
        cls._Target.DataManager(sregistry, options).process(graph)

        # Initialize the target-language runtime
        parizer.initialize(graph)

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

        # Reduce flops (no arithmetic alterations)
        clusters = cse(clusters, sregistry)

        # Lifting may create fusion opportunities, which in turn may enable
        # further optimizations
        clusters = fuse(clusters)
        clusters = eliminate_arrays(clusters)

        return clusters

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

        # GPU parallelism
        parizer = cls._Target.Parizer(sregistry, options, platform)
        parizer.make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._Target.DataManager(sregistry, options).process(graph)

        # Initialize the target-language runtime
        parizer.initialize(graph)

        # TODO: This should be moved right below the `mpiize` pass, but currently calling
        # `make_gpudirect` before Symbol definitions` block would create Blocks before
        # creating C variables. That would lead to MPI_Request variables being local to
        # their blocks. This way, it would generate incorrect C code.
        if options['gpu-direct']:
            parizer.make_gpudirect(graph)

        return graph


class DeviceCustomOperator(DeviceOperatorMixin, CustomOperator):

    @classmethod
    def _make_dsl_passes_mapper(cls, **kwargs):
        return {
            'collect-derivs': collect_derivatives,
        }

    @classmethod
    def _make_exprs_passes_mapper(cls, **kwargs):
        options = kwargs['options']

        # This callback is used by `buffering` to replace host Functions with
        # Arrays, used as device buffers for streaming-in and -out of data
        def callback(f):
            if not is_on_device(f, options['gpu-fit']):
                return [f.time_dim]
            else:
                return None

        return {
            'buffering': lambda i: buffering(i, callback, options)
        }

    @classmethod
    def _make_clusters_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        runs_on_host, reads_if_on_host = make_callbacks(options)

        return {
            'blocking': Blocking(options).process,
            'tasking': Tasker(runs_on_host).process,
            'streaming': Streaming(reads_if_on_host).process,
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

        parizer = cls._Target.Parizer(sregistry, options, platform)
        orchestrator = cls._Target.Orchestrator(sregistry)

        return {
            'optcomms': partial(optimize_halospots),
            'parallel': parizer.make_parallel,
            'orchestrate': partial(orchestrator.process),
            'mpi': partial(mpiize, mode=options['mpi']),
            'prodders': partial(hoist_prodders),
            'gpu-direct': partial(parizer.make_gpudirect),
            'init': parizer.initialize
        }

    _known_passes = (
        # DSL
        'collect-derivs',
        # Expressions
        'buffering',
        # Clusters
        'blocking', 'tasking', 'streaming', 'factorize', 'fuse', 'lift',
        'cire-sops', 'cse', 'opt-pows', 'topofuse',
        # IET
        'optcomms', 'orchestrate', 'parallel', 'mpi', 'prodders', 'gpu-direct'
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

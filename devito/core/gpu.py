from functools import partial

import numpy as np

from devito.core.operator import CoreOperator, CustomOperator, ParTile
from devito.exceptions import InvalidOperator
from devito.operator.operator import rcompile
from devito.passes import is_on_device
from devito.passes.equations import collect_derivatives
from devito.passes.clusters import (Lift, Streaming, Tasker, blocking, buffering,
                                    cire, cse, factorize, fission, fuse,
                                    optimize_pows)
from devito.passes.iet import (DeviceOmpTarget, DeviceAccTarget, mpiize, hoist_prodders,
                               linearize, pthreadify, relax_incr_dimensions)
from devito.tools import as_tuple, timed_pass

__all__ = ['DeviceNoopOperator', 'DeviceAdvOperator', 'DeviceCustomOperator',
           'DeviceNoopOmpOperator', 'DeviceAdvOmpOperator', 'DeviceFsgOmpOperator',
           'DeviceCustomOmpOperator', 'DeviceNoopAccOperator', 'DeviceAdvAccOperator',
           'DeviceFsgAccOperator', 'DeviceCustomAccOperator']


class DeviceOperatorMixin(object):

    BLOCK_LEVELS = 0
    MPI_MODES = (True, 'basic',)

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

        # CSE
        o['cse-min-cost'] = oo.pop('cse-min-cost', cls.CSE_MIN_COST)

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
        o['expand'] = oo.pop('expand', cls.EXPAND)
        o['optcomms'] = oo.pop('optcomms', True)
        o['linearize'] = oo.pop('linearize', False)
        o['mapify-reduce'] = oo.pop('mapify-reduce', cls.MAPIFY_REDUCE)
        o['index-mode'] = oo.pop('index-mode', cls.INDEX_MODE)

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

    @classmethod
    def _rcompile_wrapper(cls, **kwargs):
        options = kwargs['options']

        def wrapper(expressions, kwargs=kwargs, mode='default'):
            if mode == 'host':
                kwargs = {
                    'platform': 'cpu64',
                    'language': 'C' if options['par-disabled'] else 'openmp',
                    'compiler': 'custom',
                }
            return rcompile(expressions, kwargs)

        return wrapper

# Mode level


class DeviceNoopOperator(DeviceOperatorMixin, CoreOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        compiler = kwargs['compiler']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        mpiize(graph, **kwargs)

        # GPU parallelism
        parizer = cls._Target.Parizer(sregistry, options, platform, compiler)
        parizer.make_parallel(graph)
        parizer.initialize(graph, options=options)

        # Symbol definitions
        cls._Target.DataManager(**kwargs).process(graph)

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
        clusters = cire(clusters, 'sops', sregistry, options, platform)
        clusters = factorize(clusters)
        clusters = optimize_pows(clusters)

        # The previous passes may have created fusion opportunities
        clusters = fuse(clusters)

        # Reduce flops
        clusters = cse(clusters, sregistry, options)

        # Blocking to define thread blocks
        if options['blocklazy']:
            clusters = blocking(clusters, sregistry, options)

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        compiler = kwargs['compiler']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        mpiize(graph, **kwargs)

        # Lower BlockDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph, **kwargs)

        # GPU parallelism
        parizer = cls._Target.Parizer(sregistry, options, platform, compiler)
        parizer.make_parallel(graph)
        parizer.initialize(graph, options=options)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._Target.DataManager(**kwargs).process(graph)

        # Linearize n-dimensional Indexeds
        linearize(graph, **kwargs)

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

        # Callbacks used by `buffering`, `Tasking` and `Streaming`
        callback = lambda f: on_host(f, options)
        runs_on_host, reads_if_on_host = make_callbacks(options)

        return {
            'buffering': lambda i: buffering(i, callback, sregistry, options),
            'blocking': lambda i: blocking(i, sregistry, options),
            'tasking': Tasker(runs_on_host, sregistry).process,
            'streaming': Streaming(reads_if_on_host, sregistry).process,
            'factorize': factorize,
            'fission': fission,
            'fuse': lambda i: fuse(i, options=options),
            'lift': lambda i: Lift().process(cire(i, 'invariants', sregistry,
                                                  options, platform)),
            'cire-sops': lambda i: cire(i, 'sops', sregistry, options, platform),
            'cse': lambda i: cse(i, sregistry, options),
            'opt-pows': optimize_pows,
            'topofuse': lambda i: fuse(i, toposort=True, options=options)
        }

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        compiler = kwargs['compiler']
        sregistry = kwargs['sregistry']

        parizer = cls._Target.Parizer(sregistry, options, platform, compiler)
        orchestrator = cls._Target.Orchestrator(sregistry)

        return {
            'parallel': parizer.make_parallel,
            'orchestrate': partial(orchestrator.process),
            'pthreadify': partial(pthreadify, sregistry=sregistry),
            'mpi': partial(mpiize, **kwargs),
            'linearize': partial(linearize, **kwargs),
            'prodders': partial(hoist_prodders),
            'init': partial(parizer.initialize, options=options)
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
        'orchestrate', 'pthreadify', 'parallel', 'mpi', 'linearize', 'prodders'
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

        # Enforce linearization to mitigate LLVM issue:
        # https://github.com/llvm/llvm-project/issues/56389
        # Most OpenMP-offloading compilers are based on LLVM, and despite
        # not all of them reuse necessarily the same parloop runtime, some
        # do, or might do in the future
        oo.setdefault('linearize', True)

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

def on_host(f, options):
    # A Dimension in `f` defining an IterationSpace that definitely
    # gets executed on the host, regardless of whether it's parallel
    # or sequential
    if not is_on_device(f, options['gpu-fit']):
        return f.time_dim
    else:
        return None


def make_callbacks(options, key=None):
    """
    Options-dependent callbacks used by various compiler passes.
    """

    if key is None:
        key = lambda f: on_host(f, options)

    def runs_on_host(c):
        # The only situation in which a Cluster doesn't get offloaded to
        # the device is when it writes to a host Function
        retval = {key(f) for f in c.scope.writes} - {None}
        retval = set().union(*[d._defines for d in retval])
        return retval

    def reads_if_on_host(c):
        if not runs_on_host(c):
            retval = {key(f) for f in c.scope.reads} - {None}
            retval = set().union(*[d._defines for d in retval])
            return retval
        else:
            return set()

    return runs_on_host, reads_if_on_host

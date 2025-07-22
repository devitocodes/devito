from functools import partial

import numpy as np

from devito.core.operator import CoreOperator, CustomOperator, ParTile
from devito.exceptions import InvalidOperator
from devito.operator.operator import rcompile
from devito.passes import is_on_device, stream_dimensions
from devito.passes.equations import collect_derivatives
from devito.passes.clusters import (Lift, tasking, memcpy_prefetch, blocking,
                                    buffering, cire, cse, factorize, fission, fuse,
                                    optimize_pows)
from devito.passes.iet import (DeviceOmpTarget, DeviceAccTarget, DeviceCXXOmpTarget,
                               mpiize, hoist_prodders, linearize, pthreadify,
                               relax_incr_dimensions, check_stability)
from devito.tools import as_tuple, timed_pass

__all__ = ['DeviceNoopOperator', 'DeviceAdvOperator', 'DeviceCustomOperator',
           'DeviceNoopOmpOperator', 'DeviceAdvOmpOperator', 'DeviceFsgOmpOperator',
           'DeviceCustomOmpOperator', 'DeviceNoopAccOperator', 'DeviceAdvAccOperator',
           'DeviceFsgAccOperator', 'DeviceCustomAccOperator', 'DeviceNoopCXXOmpOperator',
           'DeviceAdvCXXOmpOperator', 'DeviceFsgCXXOmpOperator',
           'DeviceCustomCXXOmpOperator']


class DeviceOperatorMixin:

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
        o['buf-reuse'] = oo.pop('buf-reuse', None)

        # Fusion
        o['fuse-tasks'] = oo.pop('fuse-tasks', False)

        # Flops minimization
        o['cse-min-cost'] = oo.pop('cse-min-cost', cls.CSE_MIN_COST)
        o['cse-algo'] = oo.pop('cse-algo', cls.CSE_ALGO)
        o['fact-schedule'] = oo.pop('fact-schedule', cls.FACT_SCHEDULE)

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
        o['par-tile'] = ParTile(oo.pop('par-tile', False), default=(32, 4, 4),
                                sparse=oo.pop('par-tile-sparse', None),
                                reduce=oo.pop('par-tile-reduce', None))
        o['par-collapse-ncores'] = 1  # Always collapse (meaningful if `par-tile=False`)
        o['par-collapse-work'] = 1  # Always collapse (meaningful if `par-tile=False`)
        o['par-chunk-nonaffine'] = oo.pop('par-chunk-nonaffine', cls.PAR_CHUNK_NONAFFINE)
        o['par-dynamic-work'] = np.inf  # Always use static scheduling
        o['par-nested'] = np.inf  # Never use nested parallelism
        o['par-disabled'] = oo.pop('par-disabled', True)  # No host parallelism by default
        o['gpu-fit'] = cls._normalize_gpu_fit(oo, **kwargs)
        o['gpu-create'] = as_tuple(oo.pop('gpu-create', ()))

        # Distributed parallelism
        o['dist-drop-unwritten'] = oo.pop('dist-drop-unwritten', cls.DIST_DROP_UNWRITTEN)

        # Code generation options for derivatives
        o['expand'] = oo.pop('expand', cls.EXPAND)
        o['deriv-schedule'] = oo.pop('deriv-schedule', cls.DERIV_SCHEDULE)
        o['deriv-unroll'] = oo.pop('deriv-unroll', False)

        # Misc
        o['opt-comms'] = oo.pop('opt-comms', True)
        o['linearize'] = oo.pop('linearize', False)
        o['mapify-reduce'] = oo.pop('mapify-reduce', cls.MAPIFY_REDUCE)
        o['index-mode'] = oo.pop('index-mode', cls.INDEX_MODE)
        o['place-transfers'] = oo.pop('place-transfers', True)
        o['errctl'] = oo.pop('errctl', cls.ERRCTL)
        o['scalar-min-type'] = oo.pop('scalar-min-type', cls.SCALAR_MIN_TYPE)

        if oo:
            raise InvalidOperator("Unsupported optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs

    @classmethod
    def _normalize_gpu_fit(cls, oo, **kwargs):
        try:
            gfit = as_tuple(oo.pop('gpu-fit'))
            gfit = set().union(*[f.values() if f.is_AbstractTensor else [f]
                                 for f in gfit])
            return tuple(gfit)
        except KeyError:
            if any(i in kwargs['mode'] for i in ['tasking', 'streaming']):
                return (None,)
            else:
                return as_tuple(cls.GPU_FIT)

    @classmethod
    def _rcompile_wrapper(cls, **kwargs0):
        options0 = kwargs0.pop('options')

        def wrapper(expressions, mode='default', options=None, **kwargs1):
            kwargs = {**kwargs0, **kwargs1}
            options = options or {}

            if mode == 'host':
                target = {
                    'platform': 'cpu64',
                    'language': 'C' if options0['par-disabled'] else 'openmp',
                    'compiler': 'custom'
                }
            else:
                # Always use the default `par-tile` for recursive compilation
                # unless the caller explicitly overrides it so that if the user
                # supplies a multi par-tile there is no need to worry about the
                # small kernels typically generated by recursive compilation
                par_tile0 = options0['par-tile']
                par_tile = options.get('par-tile')
                if par_tile0 and par_tile:
                    options = {**options0, **options, 'par-tile': par_tile}
                elif par_tile0:
                    par_tile = ParTile(par_tile0.default, default=par_tile0.default)
                    options = {**options0, **options, 'par-tile': par_tile}
                else:
                    options = {**options0, **options}

                target = None

            return rcompile(expressions, kwargs, options, target=target)

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
        clusters = factorize(clusters, **kwargs)
        clusters = optimize_pows(clusters)

        # The previous passes may have created fusion opportunities
        clusters = fuse(clusters)

        # Reduce flops
        clusters = cse(clusters, **kwargs)

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

        # Perform error checking
        check_stability(graph, **kwargs)

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

        callback = lambda f: not is_on_device(f, options['gpu-fit'])
        stream_key = stream_wrap(callback)

        return {
            'blocking': lambda i: blocking(i, sregistry, options),
            'buffering': lambda i: buffering(i, stream_key, sregistry, options),
            'tasking': lambda i: tasking(i, stream_key, sregistry),
            'streaming': lambda i: memcpy_prefetch(i, stream_key, sregistry),
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
        orchestrator = cls._Target.Orchestrator(**kwargs)

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

class DeviceOmpOperatorMixin:

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

    @classmethod
    def _check_kwargs(cls, **kwargs):
        oo = kwargs['options']

        if len(oo['gpu-create']):
            raise InvalidOperator("Unsupported gpu-create option for omp operators")


class DeviceNoopOmpOperator(DeviceOmpOperatorMixin, DeviceNoopOperator):
    pass


class DeviceNoopCXXOmpOperator(DeviceNoopOmpOperator):
    _Target = DeviceCXXOmpTarget
    LINEARIZE = True


class DeviceAdvOmpOperator(DeviceOmpOperatorMixin, DeviceAdvOperator):
    pass


class DeviceAdvCXXOmpOperator(DeviceAdvOmpOperator):
    _Target = DeviceCXXOmpTarget
    LINEARIZE = True


class DeviceFsgOmpOperator(DeviceOmpOperatorMixin, DeviceFsgOperator):
    pass


class DeviceFsgCXXOmpOperator(DeviceFsgOmpOperator):
    _Target = DeviceCXXOmpTarget
    LINEARIZE = True


class DeviceCustomOmpOperator(DeviceOmpOperatorMixin, DeviceCustomOperator):

    _known_passes = DeviceCustomOperator._known_passes + ('openmp',)
    assert not (set(_known_passes) & set(DeviceCustomOperator._known_passes_disabled))

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        mapper = super()._make_iet_passes_mapper(**kwargs)
        mapper['openmp'] = mapper['parallel']
        return mapper


class DeviceCustomCXXOmpOperator(DeviceCustomOmpOperator):
    _Target = DeviceCXXOmpTarget
    LINEARIZE = True


# OpenACC

class DeviceAccOperatorMixin:

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


# *** Utils

def stream_wrap(callback):
    def stream_key(items, *args):
        """
        Given one or more Functions `f(d_1, ...d_n)`, return the Dimensions
        `(d_i, ..., d_n)` requiring data streaming.
        """
        found = [f for f in as_tuple(items) if callback(f)]
        retval = {stream_dimensions(f) for f in found}
        if len(retval) > 1:
            raise ValueError("Cannot determine homogenous stream Dimensions")
        elif len(retval) == 1:
            return retval.pop()
        else:
            return None

    return stream_key

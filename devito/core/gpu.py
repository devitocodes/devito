from contextlib import redirect_stdout
import io
import os
from functools import partial
from io import StringIO

import numpy as np
from devito.arch.archinfo import get_nvidia_cc

from devito.core.operator import CoreOperator, CustomOperator, ParTile

from devito.core.cpu import XdslAdvOperator, generate_pipeline

from devito.exceptions import InvalidOperator
from devito.operator.operator import rcompile
from devito.passes import is_on_device
from devito.passes.equations import collect_derivatives
from devito.passes.clusters import (Lift, Streaming, Tasker, blocking, buffering,
                                    cire, cse, factorize, fission, fuse,
                                    optimize_pows)
from devito.passes.iet import (DeviceOmpTarget, DeviceAccTarget, mpiize, hoist_prodders,
                               linearize, pthreadify, relax_incr_dimensions)
from devito.logger import info, perf
from devito.mpi import MPI

from devito.tools import as_tuple, timed_pass

from xdsl.printer import Printer
from xdsl.xdsl_opt_main import xDSLOptMain

from devito.ir.ietxdsl.cluster_to_ssa import finalize_module_with_globals

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
        o['par-tile'] = ParTile(oo.pop('par-tile', False), default=(32, 4, 4))
        o['par-collapse-ncores'] = 1  # Always collapse (meaningful if `par-tile=False`)
        o['par-collapse-work'] = 1  # Always collapse (meaningful if `par-tile=False`)
        o['par-chunk-nonaffine'] = oo.pop('par-chunk-nonaffine', cls.PAR_CHUNK_NONAFFINE)
        o['par-dynamic-work'] = np.inf  # Always use static scheduling
        o['par-nested'] = np.inf  # Never use nested parallelism
        o['par-disabled'] = oo.pop('par-disabled', True)  # No host parallelism by default
        o['gpu-fit'] = as_tuple(oo.pop('gpu-fit', cls._normalize_gpu_fit(**kwargs)))
        o['gpu-create'] = as_tuple(oo.pop('gpu-create', ()))

        # Distributed parallelism
        o['dist-drop-unwritten'] = oo.pop('dist-drop-unwritten', cls.DIST_DROP_UNWRITTEN)

        # Misc
        o['expand'] = oo.pop('expand', cls.EXPAND)
        o['optcomms'] = oo.pop('optcomms', True)
        o['linearize'] = oo.pop('linearize', False)
        o['mapify-reduce'] = oo.pop('mapify-reduce', cls.MAPIFY_REDUCE)
        o['index-mode'] = oo.pop('index-mode', cls.INDEX_MODE)
        o['place-transfers'] = oo.pop('place-transfers', True)

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

    @classmethod
    def _check_kwargs(cls, **kwargs):
        oo = kwargs['options']

        if len(oo['gpu-create']):
            raise InvalidOperator("Unsupported gpu-create option for omp operators")


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


class XdslAdvDeviceOperator(XdslAdvOperator):

    _Target = DeviceOmpTarget

    def _jit_compile(self):
        """
        JIT-compile the C code generated by the Operator.
        It is ensured that JIT compilation will only be performed
        once per Operator, reagardless of how many times this method
        is invoked.
        """
        with self._profiler.timer_on('jit-compile'):
            is_mpi = MPI.Is_initialized()
            is_gpu = os.environ.get("DEVITO_PLATFORM", None) == 'nvidiaX'

            if is_mpi and is_gpu:
                raise RuntimeError("Cannot run MPI+GPU for now!")

            # specialize the code for the specific apply parameters
            finalize_module_with_globals(self._module, self._jit_kernel_constants,
                                         gpu_boilerplate=is_gpu)

            # print module as IR
            module_str = StringIO()
            Printer(stream=module_str).print(self._module)
            module_str = module_str.getvalue()

            xdsl_pipeline = generate_XDSL_GPU_PIPELINE()
            # Get GPU blocking shapes
            block_sizes: list[int] = [min(target, self._jit_kernel_constants.get(f"{dim}_size", 1)) for target, dim in zip([32, 4, 8], ["x", "y", "z"])]  # noqa
            block_sizes = ','.join(str(bs) for bs in block_sizes)
            mlir_pipeline = generate_MLIR_GPU_PIPELINE(block_sizes)

            # allow jit backdooring to provide your own xdsl code
            backdoor = os.getenv('XDSL_JIT_BACKDOOR')
            if backdoor is not None:
                if os.path.splitext(backdoor)[1] == ".so":
                    info(f"JIT Backdoor: skipping compilation and using {backdoor}")
                    self._tf.name = backdoor
                    return
                print("JIT Backdoor: loading xdsl file from: " + backdoor)
                with open(backdoor, 'r') as f:
                    module_str = f.read()

            # Uncomment to print the module_str
            # Printer().print(module_str)
            source_name = os.path.splitext(self._tf.name)[0] + ".mlir"
            source_file = open(source_name, "w")
            source_file.write(module_str)
            source_file.close()

            # Compile IR using xdsl-opt | mlir-opt | mlir-translate | clang
            cflags = "-O3 -march=native -mtune=native -lmlir_c_runner_utils"

            try:
                cc = "clang"

                cflags += " -lmlir_cuda_runtime "
                cflags += " -shared "

                # TODO More detailed error handling manually,
                # instead of relying on a bash-only feature.

                # xdsl-opt, get xDSL IR
                # TODO: Remove quotes in pipeline; currently workaround with [1:-1]
                xdsl_args=[source_name, "--allow-unregistered-dialect", "-p", xdsl_pipeline[1:-1]+f',mlir-opt{{arguments=--mlir-print-op-generic,--allow-unregistered-dialect,-p,{mlir_pipeline}}}']
                xdsl = xDSLOptMain(args=xdsl_args)
                out = io.StringIO()
                perf("-----------------")
                perf(f"xdsl-opt {' '.join(xdsl_args)}")
                with redirect_stdout(out):
                    xdsl.run()

                # mlir-opt
                # mlir_cmd = f'mlir-opt -p {mlir_pipeline}'
                # out = self.compile(mlir_cmd, out.getvalue())

                # Printer().print(out)

                mlir_translate_cmd = 'mlir-translate --mlir-to-llvmir'
                out = self.compile(mlir_translate_cmd, out.getvalue())
                # Printer().print(out)

                # Compile with clang and get LLVM-IR
                clang_cmd = f'{cc} {cflags} -o {self._tf.name} {self._interop_tf.name} -xir -'  # noqa
                out = self.compile(clang_cmd, out)

            except Exception as ex:
                print("error")
                raise ex

        elapsed = self._profiler.py_timers['jit-compile']

        perf("XDSLAdvDeviceOperator `%s` jit-compiled `%s` in %.2f s with `mlir-opt`" %
             (self.name, source_name, elapsed))


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


def generate_XDSL_GPU_PIPELINE():
    passes = [
        "stencil-shape-inference",
        "convert-stencil-to-ll-mlir{target=gpu}",
        "reconcile-unrealized-casts",
        "printf-to-llvm",
        "canonicalize"
    ]

    return generate_pipeline(passes)


# gpu-launch-sink-index-computations seemed to have no impact
def generate_MLIR_GPU_PIPELINE(block_sizes):
    passes = [
        "builtin.module(test-math-algebraic-simplification",
        f"scf-parallel-loop-tiling{{parallel-loop-tile-sizes={block_sizes}}}",
        "func.func(gpu-map-parallel-loops)",
        "convert-parallel-loops-to-gpu",
        "lower-affine",
        "canonicalize",
        "cse",
        "fold-memref-alias-ops",
        "gpu-launch-sink-index-computations",
        "gpu-kernel-outlining",
        "canonicalize{region-simplify}",
        "cse",
        "fold-memref-alias-ops",
        "expand-strided-metadata",
        "lower-affine",
        "canonicalize",
        "cse",
        "func.func(gpu-async-region)",
        "canonicalize",
        "cse",
        "convert-arith-to-llvm{index-bitwidth=64}",
        "convert-scf-to-cf",
        "convert-cf-to-llvm{index-bitwidth=64}",
        "canonicalize",
        "cse",
        "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
        f"nvvm-attach-target{{O=3 ftz fast chip=sm_{get_nvidia_cc()}}}",
        "gpu.module(convert-gpu-to-nvvm,canonicalize,cse)",
        "gpu-to-llvm",
        "gpu-module-to-binary",
        "canonicalize",
        "cse)"
    ]

    return generate_pipeline(passes)

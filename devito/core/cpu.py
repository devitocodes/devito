import os
import subprocess
import ctypes
import tempfile

from io import StringIO
from collections import OrderedDict

from functools import partial

from devito.core.operator import CoreOperator, CustomOperator, ParTile
from devito.exceptions import InvalidOperator
from devito.passes.equations import collect_derivatives
from devito.passes.clusters import (Lift, blocking, buffering, cire, cse,
                                    factorize, fission, fuse, optimize_pows,
                                    optimize_hyperplanes)
from devito.passes.iet import (CTarget, OmpTarget, avoid_denormals, linearize, mpiize,
                               hoist_prodders, relax_incr_dimensions)
from devito.mpi import MPI
from devito.tools import timed_pass

from devito.logger import info, perf
from devito.operator.profiling import create_profile
from devito.ir.iet import Callable, MetaCall

from devito.tools import flatten, filter_sorted, OrderedSet

from devito.ir.ietxdsl.cluster_to_ssa import (ExtractDevitoStencilConversion,
                                              convert_devito_stencil_to_xdsl_stencil,
                                              finalize_module_with_globals)  # noqa

from devito.types import TimeFunction
from devito.types.mlir_types import ptr_of, f32

from xdsl.printer import Printer


__all__ = ['Cpu64NoopCOperator', 'Cpu64NoopOmpOperator', 'Cpu64AdvCOperator',
           'Cpu64AdvOmpOperator', 'Cpu64FsgCOperator', 'Cpu64FsgOmpOperator',
           'Cpu64CustomOperator', 'XdslnoopOperator', 'XdslAdvOperator']


class Cpu64OperatorMixin(object):

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        o = {}
        oo = kwargs['options']

        # Execution modes
        o['openmp'] = oo.pop('openmp')
        o['mpi'] = oo.pop('mpi')
        o['parallel'] = o['openmp']  # Backwards compatibility

        # Buffering
        o['buf-async-degree'] = oo.pop('buf-async-degree', None)

        # Fusion
        o['fuse-tasks'] = oo.pop('fuse-tasks', False)

        # CSE
        o['cse-min-cost'] = oo.pop('cse-min-cost', cls.CSE_MIN_COST)

        # Blocking
        o['blockinner'] = oo.pop('blockinner', False)
        o['blocklevels'] = oo.pop('blocklevels', cls.BLOCK_LEVELS)
        o['blockeager'] = oo.pop('blockeager', cls.BLOCK_EAGER)
        o['blocklazy'] = oo.pop('blocklazy', not o['blockeager'])
        o['blockrelax'] = oo.pop('blockrelax', cls.BLOCK_RELAX)
        o['skewing'] = oo.pop('skewing', False)
        o['par-tile'] = ParTile(oo.pop('par-tile', False), default=16)

        # CIRE
        o['min-storage'] = oo.pop('min-storage', False)
        o['cire-rotate'] = oo.pop('cire-rotate', False)
        o['cire-maxpar'] = oo.pop('cire-maxpar', False)
        o['cire-ftemps'] = oo.pop('cire-ftemps', False)
        o['cire-mingain'] = oo.pop('cire-mingain', cls.CIRE_MINGAIN)
        o['cire-schedule'] = oo.pop('cire-schedule', cls.CIRE_SCHEDULE)

        # Shared-memory parallelism
        o['par-collapse-ncores'] = oo.pop('par-collapse-ncores', cls.PAR_COLLAPSE_NCORES)
        o['par-collapse-work'] = oo.pop('par-collapse-work', cls.PAR_COLLAPSE_WORK)
        o['par-chunk-nonaffine'] = oo.pop('par-chunk-nonaffine', cls.PAR_CHUNK_NONAFFINE)
        o['par-dynamic-work'] = oo.pop('par-dynamic-work', cls.PAR_DYNAMIC_WORK)
        o['par-nested'] = oo.pop('par-nested', cls.PAR_NESTED)

        # Distributed parallelism
        o['dist-drop-unwritten'] = oo.pop('dist-drop-unwritten', cls.DIST_DROP_UNWRITTEN)

        # Misc
        o['expand'] = oo.pop('expand', cls.EXPAND)
        o['optcomms'] = oo.pop('optcomms', True)
        o['linearize'] = oo.pop('linearize', False)
        o['mapify-reduce'] = oo.pop('mapify-reduce', cls.MAPIFY_REDUCE)
        o['index-mode'] = oo.pop('index-mode', cls.INDEX_MODE)
        o['place-transfers'] = oo.pop('place-transfers', True)

        # Recognised but unused by the CPU backend
        oo.pop('par-disabled', None)
        oo.pop('gpu-fit', None)
        oo.pop('gpu-create', None)

        if oo:
            raise InvalidOperator("Unrecognized optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs


# Mode level

class Cpu64NoopOperator(Cpu64OperatorMixin, CoreOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        compiler = kwargs['compiler']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        mpiize(graph, **kwargs)

        # Shared-memory parallelism
        if options['openmp']:
            parizer = cls._Target.Parizer(sregistry, options, platform, compiler)
            parizer.make_parallel(graph)
            parizer.initialize(graph, options=options)

        # Symbol definitions
        cls._Target.DataManager(**kwargs).process(graph)

        return graph


class Cpu64AdvOperator(Cpu64OperatorMixin, CoreOperator):

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

        # Blocking to improve data locality
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

        # Blocking to improve data locality
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

        # Flush denormal numbers
        avoid_denormals(graph, platform=platform)

        # Distributed-memory parallelism
        mpiize(graph, **kwargs)

        # Lower BlockDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph, **kwargs)

        # Parallelism
        parizer = cls._Target.Parizer(sregistry, options, platform, compiler)
        parizer.make_simd(graph)
        parizer.make_parallel(graph)
        parizer.initialize(graph, options=options)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._Target.DataManager(**kwargs).process(graph)

        # Linearize n-dimensional Indexeds
        linearize(graph, **kwargs)

        return graph


class XdslnoopOperator(Cpu64OperatorMixin, CoreOperator):

    _Target = CTarget

    @classmethod
    def _build(cls, expressions, **kwargs):

        # Lots of duplicate code, to drop
        perf("Building an xDSL operator")
        # Python- (i.e., compile-) and C-level (i.e., run-time) performance
        profiler = create_profile('timers')

        # Lower the input expressions into an IET and a module. iet is not used
        perf("Lower expressions to a module")
        irs, byproduct = cls._lower(expressions, profiler=profiler, **kwargs)

        # Make it an actual Operator
        op = Callable.__new__(cls, **irs.iet.args)
        Callable.__init__(op, **op.args)

        # Header files, etc.
        op._headers = OrderedSet(*cls._default_headers)
        op._headers.update(byproduct.headers)
        op._globals = OrderedSet(*cls._default_globals)
        op._includes = OrderedSet(*cls._default_includes)
        op._includes.update(profiler._default_includes)
        op._includes.update(byproduct.includes)

        # Required for the jit-compilation
        op._compiler = kwargs['compiler']
        op._language = kwargs['language']
        op._lib = None
        op._cfunction = None

        # Potentially required for lazily allocated Functions
        op._mode = kwargs['mode']
        op._options = kwargs['options']
        op._allocator = kwargs['allocator']
        op._platform = kwargs['platform']

        # References to local or external routines
        op._func_table = OrderedDict()
        op._func_table.update(OrderedDict([(i, MetaCall(None, False))
                                           for i in profiler._ext_calls]))
        op._func_table.update(OrderedDict([(i.root.name, i) for i in byproduct.funcs]))

        # Internal mutable state to store information about previous runs, autotuning
        # reports, etc
        op._state = cls._initialize_state(**kwargs)

        # Produced by the various compilation passes

        op._reads = filter_sorted(flatten(e.reads for e in irs.expressions))
        op._writes = filter_sorted(flatten(e.writes for e in irs.expressions))
        op._dimensions = set().union(*[e.dimensions for e in irs.expressions])
        op._dtype, op._dspace = irs.clusters.meta
        op._profiler = profiler

        module = cls._lower_stencil(irs.expressions)
        op._module = module

        return op

    @classmethod
    def _lower_stencil(cls, expressions):
        # [Eq] -> [xdsl]
        # Lower expressions to a builtin.ModuleOp
        conv = ExtractDevitoStencilConversion(expressions)
        module = conv.convert()
        # Uncomment to print
        # Printer().print(module)
        convert_devito_stencil_to_xdsl_stencil(module, timed=True)
        # Uncomment to print
        # Printer().print(module)
        return module

    @property
    def mpi_shape(self) -> tuple:
        # TODO: move it elsewhere
        dist = self.functions[0].grid.distributor

        # reverse topology for row->column major
        return dist.topology, dist.myrank

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
            is_omp = os.environ.get("DEVITO_LANGUAGE", None) == 'openmp'

            if is_mpi and is_gpu:
                raise RuntimeError("Cannot run MPI+GPU for now!")

            if is_omp and is_gpu:
                raise RuntimeError("Cannot run OMP+GPU!")

            # specialize the code for the specific apply parameters
            finalize_module_with_globals(self._module, self._jit_kernel_constants,
                                         gpu_boilerplate=is_gpu)

            # print module as IR
            module_str = StringIO()
            Printer(stream=module_str).print(self._module)
            module_str = module_str.getvalue()

            xdsl_pipeline = generate_XDSL_CPU_noop_PIPELINE()
            mlir_pipeline = generate_MLIR_CPU_noop_PIPELINE()

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

                if is_mpi:
                    cflags += ' -lmpi '
                    cc = "mpicc -cc=clang"
                if is_omp:
                    cflags += " -fopenmp "
                if is_gpu:
                    cflags += " -lmlir_cuda_runtime "

                cflags += " -shared "

                # TODO More detailed error handling manually,
                # instead of relying on a bash-only feature.

                # xdsl-opt, get xDSL IR
                xdsl_cmd = f'xdsl-opt {source_name} -p {xdsl_pipeline}'
                out = self.compile(xdsl_cmd)
                # Printer().print(out)

                # mlir-opt
                mlir_cmd = f'mlir-opt -p {mlir_pipeline}'
                out = self.compile(mlir_cmd, out)
                #  Printer().print(out)

                mlir_translate_cmd = 'mlir-translate --mlir-to-llvmir'
                out = self.compile(mlir_translate_cmd, out)
                # Printer().print(out)

                # Compile with clang and get LLVM-IR
                clang_cmd = f'{cc} {cflags} -o {self._tf.name} {self._interop_tf.name} -xir -'  # noqa
                out = self.compile(clang_cmd, out)

            except Exception as ex:
                print("error")
                raise ex

        elapsed = self._profiler.py_timers['jit-compile']

        perf("XDSLOperator `%s` jit-compiled `%s` in %.2f s with `mlir-opt`" %
             (self.name, source_name, elapsed))

    def _cmd_compile(self, cmd, input=None):
        # Could be dropped unless PIPE is never empty in the future
        stdin = subprocess.PIPE if input is not None else None  # noqa

        res = subprocess.run(
            cmd,
            input=input,
            shell=True,
            text=True,
            capture_output=True,
            executable="/bin/bash"
        )

        if res.returncode != 0:
            print("compilation failed with output:")
            print(res.stderr)

        assert res.returncode == 0
        return res.returncode, res.stdout, res.stderr

    def apply(self, **kwargs):
        # Build the arguments list to invoke the kernel function
        with self._profiler.timer_on('arguments'):
            args = self.arguments(**kwargs)
            self._jit_kernel_constants = args

        cfunction = self.cfunction
        try:
            # Invoke kernel function with args
            arg_values = self._construct_cfunction_args(args)
            with self._profiler.timer_on('apply', comm=args.comm):
                cfunction(*arg_values)
        except ctypes.ArgumentError as e:
            if e.args[0].startswith("argument "):
                argnum = int(e.args[0][9:].split(':')[0]) - 1
                newmsg = "error in argument '%s' with value '%s': %s" % (
                    self.parameters[argnum].name,
                    arg_values[argnum],
                    e.args[0])
                raise ctypes.ArgumentError(newmsg) from e
            else:
                raise

        # Post-process runtime arguments
        self._postprocess_arguments(args, **kwargs)

        # Output summary of performance achieved
        return self._emit_apply_profiling(args)

    @property
    def cfunction(self):
        """The JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:

            delete = not os.getenv("XDSL_SKIP_CLEAN", False)
            self._tf = tempfile.NamedTemporaryFile(prefix="devito-jit-", suffix='.so',
                                                   delete=delete)
            self._interop_tf = tempfile.NamedTemporaryFile(prefix="devito-jit-interop-",
                                                           suffix=".o", delete=delete)
            self._make_interop_o()
            self._jit_compile()
            self.setup_memref_args()
            self._lib = self._compiler.load(self._tf.name)
            self._lib.name = self._tf.name

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, "apply_kernel")
            # Associate a C type to each argument for runtime type check
            argtypes = self._construct_cfunction_args(self._jit_kernel_constants,
                                                      get_types=True)
            self._cfunction.argtypes = argtypes

        return self._cfunction

    def _make_interop_o(self):
        """
        compile the interop.o file
        """
        res = subprocess.run(
            f'clang -x c - -c -o {self._interop_tf.name}',
            shell=True,
            input=_INTEROP_C,
            text=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        assert res.returncode == 0

    def compile(self, cmd, stdout=None):
        # Execute each command and store the outputs
        outputs = []
        return_code, stdout, stderr = self._cmd_compile(cmd, stdout)
        # Use DEVITO_LOGGING=DEBUG to print
        perf("-----------------")
        perf(cmd)
        outputs.append({
            'command': cmd,
            'return_code': return_code,
            'stdout': stdout,
            'stderr': stderr
        })

        return stdout

    def setup_memref_args(self):
        """
        Add memrefs to args dictionary so they can be passed to the cfunction
        """
        args = dict()
        for arg in self.functions:
            # For every TimeFunction add memref
            if isinstance(arg, TimeFunction):
                data = arg._data
                for t in range(data.shape[0]):
                    args[f'{arg._C_name}_{t}'] = data[t, ...].ctypes.data_as(ptr_of(f32))

        self._jit_kernel_constants.update(args)

    def _construct_cfunction_args(self, args, get_types=False):
        """
        Either construct the args for the cfunction, or construct the
        arg types for it.
        """
        ps = {
            p._C_name: p._C_ctype for p in self.parameters
        }

        things = []
        things_types = []

        for name in get_arg_names_from_module(self._module):
            thing = args[name]
            things.append(thing)
            if name in ps:
                things_types.append(ps[name])
            else:
                things_types.append(type(thing))

        if get_types:
            return things_types
        else:
            return things


class XdslAdvOperator(XdslnoopOperator):

    _Target = OmpTarget

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
            is_omp = os.environ.get("DEVITO_LANGUAGE", None) == 'openmp'

            if is_mpi and is_gpu:
                raise RuntimeError("Cannot run MPI+GPU for now!")

            if is_omp and is_gpu:
                raise RuntimeError("Cannot run OMP+GPU!")

            # specialize the code for the specific apply parameters
            finalize_module_with_globals(self._module, self._jit_kernel_constants,
                                         gpu_boilerplate=is_gpu)

            # print module as IR
            module_str = StringIO()
            Printer(stream=module_str).print(self._module)
            module_str = module_str.getvalue()

            to_tile = len(list(filter(lambda d: d.is_Space, self.dimensions)))-1

            xdsl_pipeline = generate_XDSL_CPU_PIPELINE(to_tile)

            mlir_pipeline = generate_MLIR_CPU_PIPELINE()

            if is_omp:
                mlir_pipeline = generate_MLIR_OPENMP_PIPELINE()

            if is_mpi:
                shape, _ = self.mpi_shape
                # Run with restrict domain=false so we only introduce the swaps but don't
                # reduce the domain of the computation
                # (as devito has already done that for us)
                slices = ','.join(str(x) for x in shape)

                decomp = "2d-grid" if len(shape) == 2 else "3d-grid"

                decomp = f"{{strategy={decomp} slices={slices} restrict_domain=false}}"
                xdsl_pipeline = generate_XDSL_MPI_PIPELINE(decomp, to_tile)

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

                if is_mpi:
                    cflags += ' -lmpi '
                    cc = "mpicc -cc=clang"
                if is_omp:
                    cflags += " -fopenmp "
                if is_gpu:
                    cflags += " -lmlir_cuda_runtime "

                cflags += " -shared "

                # TODO More detailed error handling manually,
                # instead of relying on a bash-only feature.

                # xdsl-opt, get xDSL IR
                xdsl_cmd = f'xdsl-opt {source_name} -p {xdsl_pipeline}'
                out = self.compile(xdsl_cmd)
                # Printer().print(out)

                # mlir-opt
                mlir_cmd = f'mlir-opt -p {mlir_pipeline}'
                out = self.compile(mlir_cmd, out)

                # Printer().print(out)

                mlir_translate_cmd = 'mlir-translate --mlir-to-llvmir'
                out = self.compile(mlir_translate_cmd, out)
                # Printer().print(out)

                # Compile with clang and get LLVM-IR
                clang_cmd = f'{cc} {cflags} -o {self._tf.name} {self._interop_tf.name} -xir -'  # noqa
                out = self.compile(clang_cmd, out)

            except Exception as ex:
                print("error")
                raise ex

        elapsed = self._profiler.py_timers['jit-compile']

        perf("XDSLOperator `%s` jit-compiled `%s` in %.2f s with `mlir-opt`" %
             (self.name, source_name, elapsed))


class Cpu64FsgOperator(Cpu64AdvOperator):

    """
    Operator with performance optimizations tailored "For small grids" ("Fsg").
    """

    BLOCK_EAGER = False

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        kwargs = super()._normalize_kwargs(**kwargs)

        if kwargs['options']['min-storage']:
            raise InvalidOperator('You should not use `min-storage` with `advanced-fsg '
                                  ' as they work in opposite directions')

        return kwargs


class Cpu64CustomOperator(Cpu64OperatorMixin, CustomOperator):

    _Target = OmpTarget

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

        # Callback used by `buffering`; it mimics `is_on_device`, which is used
        # on device backends
        def callback(f):
            if f.is_TimeFunction and f.save is not None:
                return f.time_dim
            else:
                return None

        return {
            'buffering': lambda i: buffering(i, callback, sregistry, options),
            'blocking': lambda i: blocking(i, sregistry, options),
            'factorize': factorize,
            'fission': fission,
            'fuse': lambda i: fuse(i, options=options),
            'lift': lambda i: Lift().process(cire(i, 'invariants', sregistry,
                                                  options, platform)),
            'cire-sops': lambda i: cire(i, 'sops', sregistry, options, platform),
            'cse': lambda i: cse(i, sregistry, options),
            'opt-pows': optimize_pows,
            'opt-hyperplanes': optimize_hyperplanes,
            'topofuse': lambda i: fuse(i, toposort=True, options=options)
        }

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        compiler = kwargs['compiler']
        sregistry = kwargs['sregistry']

        parizer = cls._Target.Parizer(sregistry, options, platform, compiler)

        return {
            'denormals': avoid_denormals,
            'blocking': partial(relax_incr_dimensions, **kwargs),
            'parallel': parizer.make_parallel,
            'openmp': parizer.make_parallel,
            'mpi': partial(mpiize, **kwargs),
            'linearize': partial(linearize, **kwargs),
            'simd': partial(parizer.make_simd),
            'prodders': hoist_prodders,
            'init': partial(parizer.initialize, options=options)
        }

    _known_passes = (
        # DSL
        'collect-derivs',
        # Expressions
        'buffering',
        # Clusters
        'blocking', 'topofuse', 'fission', 'fuse', 'factorize', 'cire-sops',
        'cse', 'lift', 'opt-pows', 'opt-hyperplanes',
        # IET
        'denormals', 'openmp', 'mpi', 'linearize', 'simd', 'prodders',
    )
    _known_passes_disabled = ('tasking', 'streaming', 'openacc')
    assert not (set(_known_passes) & set(_known_passes_disabled))


# Language level


class Cpu64NoopCOperator(Cpu64NoopOperator):
    _Target = CTarget


class Cpu64NoopOmpOperator(Cpu64NoopOperator):
    _Target = OmpTarget


class Cpu64AdvCOperator(Cpu64AdvOperator):
    _Target = CTarget


class Cpu64AdvOmpOperator(Cpu64AdvOperator):
    _Target = OmpTarget


class Cpu64FsgCOperator(Cpu64FsgOperator):
    _Target = CTarget


class Cpu64FsgOmpOperator(Cpu64FsgOperator):
    _Target = OmpTarget


# -----------XDSL
# This is a collection of xDSL optimization pipelines
# Ideally they should follow the same type of subclassing as the rest of
# the Devito Operatos

def generate_MLIR_CPU_PIPELINE():
    passes = [
        "builtin.module(canonicalize",
        "cse",
        "loop-invariant-code-motion",
        "canonicalize",
        "cse",
        "loop-invariant-code-motion",
        "cse",
        "canonicalize",
        "fold-memref-alias-ops",
        "expand-strided-metadata",
        "loop-invariant-code-motion",
        "lower-affine",
        "convert-scf-to-cf",
        "convert-math-to-llvm",
        "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
        "finalize-memref-to-llvm",
        "canonicalize",
        "cse)"
    ]

    return generate_pipeline(passes)


def generate_MLIR_CPU_noop_PIPELINE():
    passes = [
        "builtin.module(canonicalize",
        "cse",
        # "remove-dead-values",
        "canonicalize",
        "expand-strided-metadata",
        "convert-scf-to-cf",
        "convert-math-to-llvm",
        "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
        "finalize-memref-to-llvm",
        "canonicalize)",
    ]

    return generate_pipeline(passes)


def generate_MLIR_OPENMP_PIPELINE():
    passes = [
        "builtin.module(canonicalize",
        "cse",
        "loop-invariant-code-motion",
        "canonicalize",
        "cse",
        "loop-invariant-code-motion",
        "cse",
        "canonicalize",
        "fold-memref-alias-ops",
        "expand-strided-metadata",
        "loop-invariant-code-motion",
        "lower-affine",
        # "finalize-memref-to-llvm",
        # "loop-invariant-code-motion",
        # "canonicalize",
        # "cse",
        "convert-scf-to-openmp",
        "finalize-memref-to-llvm",
        "convert-scf-to-cf",
        "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
        "convert-openmp-to-llvm",
        "convert-math-to-llvm",
        # "reconcile-unrealized-casts",
        "canonicalize",
        # "print-ir",
        "cse)"
    ]

    return generate_pipeline(passes)


def generate_XDSL_CPU_PIPELINE(nb_tiled_dims):
    passes = [
        "stencil-shape-inference",
        f"convert-stencil-to-ll-mlir{{{generate_tiling_arg(nb_tiled_dims)}}}",
        "printf-to-llvm",
        "canonicalize"
    ]

    return generate_pipeline(passes)


def generate_XDSL_CPU_noop_PIPELINE():
    passes = [
        "stencil-shape-inference",
        "convert-stencil-to-ll-mlir",
        "printf-to-llvm"
    ]

    return generate_pipeline(passes)


def generate_XDSL_MPI_PIPELINE(decomp, nb_tiled_dims):
    passes = [
        f"distribute-stencil{decomp}",
        "canonicalize-dmp",
        f"convert-stencil-to-ll-mlir{{{generate_tiling_arg(nb_tiled_dims)}}}",
        "dmp-to-mpi{mpi_init=false}",
        "lower-mpi",
        "printf-to-llvm",
        "canonicalize"
    ]

    return generate_pipeline(passes)


def generate_pipeline(passes):
    passes_string = ",".join(passes)
    return f'"{passes_string}"'


# small interop shim script for stuff that we don't want to implement in mlir-ir
_INTEROP_C = """
#include <time.h>

double timer_start() {
  // return a number representing the current point in time
  // it might be offset by a fixed ammount
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (t.tv_sec) + (t.tv_nsec * 1e-9);
}

double timer_end(double start) {
  // return time elaspes since start in seconds
  return (timer_start() - start);
}
"""


def generate_tiling_arg(nb_tiled_dims: int):
    """
    Generate the tile-sizes arg for the convert-stencil-to-ll-mlir pass.
    Generating no argument if the diled_dims arg is 0
    """
    if nb_tiled_dims == 0:
        return ''
    return "tile-sizes=" + ",".join(["64"]*nb_tiled_dims)


def get_arg_names_from_module(op):
    return [
        str_attr.data for str_attr in op.body.block.ops.first.attributes['param_names'].data  # noqa
    ]

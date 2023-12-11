import os
import subprocess
import ctypes
import tempfile

from math import ceil
from collections import OrderedDict, namedtuple
from io import StringIO
from operator import attrgetter

from cached_property import cached_property

from devito import Operator
from devito.arch import compiler_registry, platform_registry
from devito.data import default_allocator
from devito.exceptions import InvalidOperator
from devito.ir.clusters import ClusterGroup, clusterize
from devito.ir.equations import LoweredEq, lower_exprs
from devito.ir.iet import (Callable, CInterface, EntryFunction, FindSymbols, MetaCall,
                           derive_parameters, iet_build)
from devito.ir.ietxdsl import (finalize_module_with_globals)
from devito.ir.stree import stree_build
from devito.ir.support import AccessMode, SymbolRegistry
from devito.ir.ietxdsl.cluster_to_ssa import (ExtractDevitoStencilConversion,
                                              convert_devito_stencil_to_xdsl_stencil)
from devito.logger import debug, info, perf, warning, is_log_enabled_for
from devito.operator.operator import IRs
from devito.operator.profiling import AdvancedProfilerVerbose, create_profile
from devito.parameters import configuration
from devito.passes import (Graph, lower_index_derivatives, generate_implicit,
                           generate_macros, minimize_symbols, unevaluate)
from devito.passes.iet import CTarget
from devito.symbolics import estimate_cost
from devito.tools import (DAG, OrderedSet, ReducerMap, as_tuple, flatten,
                          filter_sorted, frozendict, is_integer, split, timed_pass,
                          contains_val)
from devito.types import Evaluable, TimeFunction, Grid
from devito.types.mlir_types import ptr_of, f32
from devito.mpi import MPI

from xdsl.printer import Printer

# flake8: noqa

__all__ = ['XDSLOperator']


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
    Generate the tile-sizes arg for the convert-stencil-to-ll-mlir pass. Generating no argument if the diled_dims arg is 0
    """
    if nb_tiled_dims == 0:
        return ''
    return "tile-sizes=" + ",".join(["64"]*nb_tiled_dims)


CFLAGS = "-O3 -march=native -mtune=native -lmlir_c_runner_utils"

MLIR_CPU_PIPELINE = '"builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,expand-strided-metadata, loop-invariant-code-motion,lower-affine,convert-scf-to-cf,convert-math-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv},finalize-memref-to-llvm,canonicalize,cse)"'
MLIR_OPENMP_PIPELINE = '"builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,expand-strided-metadata, loop-invariant-code-motion,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,convert-scf-to-openmp,finalize-memref-to-llvm,convert-scf-to-cf,convert-func-to-llvm{use-bare-ptr-memref-call-conv},convert-openmp-to-llvm,convert-math-to-llvm,reconcile-unrealized-casts,canonicalize,cse)"'
# gpu-launch-sink-index-computations seemed to have no impact
MLIR_GPU_PIPELINE = lambda block_sizes: f'"builtin.module(test-math-algebraic-simplification,scf-parallel-loop-tiling{{parallel-loop-tile-sizes={block_sizes}}},func.func(gpu-map-parallel-loops),convert-parallel-loops-to-gpu,lower-affine, canonicalize,cse, fold-memref-alias-ops, gpu-launch-sink-index-computations, gpu-kernel-outlining, canonicalize{{region-simplify}},cse,fold-memref-alias-ops,expand-strided-metadata,lower-affine,canonicalize,cse,func.func(gpu-async-region),canonicalize,cse,convert-arith-to-llvm{{index-bitwidth=64}},convert-scf-to-cf,convert-cf-to-llvm{{index-bitwidth=64}},canonicalize,cse,convert-func-to-llvm{{use-bare-ptr-memref-call-conv}},nvvm-attach-target{{O=3 ftz fast}},gpu.module(convert-gpu-to-nvvm,canonicalize,cse),gpu-to-llvm,gpu-module-to-binary,canonicalize,cse)"'

XDSL_CPU_PIPELINE = lambda nb_tiled_dims: f'"stencil-shape-inference,convert-stencil-to-ll-mlir{{{generate_tiling_arg(nb_tiled_dims)}}},printf-to-llvm,canonicalize"'
XDSL_GPU_PIPELINE = "stencil-shape-inference,convert-stencil-to-ll-mlir{target=gpu},reconcile-unrealized-casts,printf-to-llvm,canonicalize"
XDSL_MPI_PIPELINE = lambda decomp, nb_tiled_dims: f'"distribute-stencil{decomp},canonicalize-dmp,convert-stencil-to-ll-mlir{{{generate_tiling_arg(nb_tiled_dims)}}},dmp-to-mpi{{mpi_init=false}},lower-mpi,printf-to-llvm,canonicalize"'


class XDSLOperator(Operator):

    _Target = CTarget

    def __new__(cls, expressions, **kwargs):
        self = super(XDSLOperator, cls).__new__(cls, expressions, **kwargs)
        delete = not os.getenv("XDSL_SKIP_CLEAN", False)
        self._tf = tempfile.NamedTemporaryFile(prefix="devito-jit-", suffix='.so',
                                               delete=delete)
        self._interop_tf = tempfile.NamedTemporaryFile(prefix="devito-jit-interop-",
                                                       suffix=".o", delete=delete)
        self._make_interop_o()
        self.__class__ = cls
        return self

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

    @property
    def mpi_shape(self) -> tuple:
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

            to_tile = len(list(filter(lambda s: str(s) in ["x", "y", "z"], self.dimensions)))-1

            xdsl_pipeline = XDSL_CPU_PIPELINE(to_tile)
            mlir_pipeline = MLIR_CPU_PIPELINE

            block_sizes: list[int] = [min(target, self._jit_kernel_constants.get(f"{dim}_size", 1)) for target, dim in zip([32, 4, 8], ["x", "y", "z"])]
            block_sizes = ','.join(str(bs) for bs in block_sizes)

            if is_omp:
                mlir_pipeline = MLIR_OPENMP_PIPELINE

            if is_mpi:
                shape, mpi_rank = self.mpi_shape
                # Run with restrict domain=false so we only introduce the swaps but don't
                # reduce the domain of the computation (as devito has already done that for us)
                slices = ','.join(str(x) for x in shape)

                decomp = "2d-grid" if len(shape) == 2 else "3d-grid"

                decomp = f"{{strategy={decomp} slices={slices} restrict_domain=false}}"
                xdsl_pipeline = XDSL_MPI_PIPELINE(decomp, to_tile)
            elif is_gpu:
                xdsl_pipeline = XDSL_GPU_PIPELINE
                mlir_pipeline = MLIR_GPU_PIPELINE(block_sizes)

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
            source_name = os.path.splitext(self._tf.name)[0] + ".mlir"
            source_file = open(source_name, "w")
            source_file.write(module_str)
            source_file.close()

            # Compile IR using xdsl-opt | mlir-opt | mlir-translate | clang
            try:
                cflags = CFLAGS
                cc = "clang"

                if is_mpi:
                    cflags += ' -lmpi '
                    cc = "mpicc -cc=clang"
                if is_omp:
                    cflags += " -fopenmp "
                if is_gpu:
                    cflags += " -lmlir_cuda_runtime "

                # TODO More detailed error handling manually,
                # instead of relying on a bash-only feature.

                xdsl_cmd = f'xdsl-opt {source_name} -p {xdsl_pipeline}'
                mlir_cmd = f'mlir-opt -p {mlir_pipeline}'
                mlir_translate_cmd = 'mlir-translate --mlir-to-llvmir'
                clang_cmd = f'{cc} {cflags} -shared -o {self._tf.name} {self._interop_tf.name} -xir -'


                comp_steps = [
                              xdsl_cmd,
                              mlir_cmd,
                              mlir_translate_cmd,
                              clang_cmd 
                             ]

                # Execute each command and store the outputs
                outputs = []
                stdout = None
                for cmd in comp_steps:
                    return_code, stdout, stderr = self._cmd_compile(cmd, stdout)
                    # Use DEVITO_LOGGING=DEBUG to print
                    debug(cmd)
                    outputs.append({
                        'command': cmd,
                        'return_code': return_code,
                        'stdout': stdout,
                        'stderr': stderr
                    })
                
            except Exception as ex:
                print("error")
                raise ex

        elapsed = self._profiler.py_timers['jit-compile']

        perf("XDSLOperator `%s` jit-compiled `%s` in %.2f s with `mlir-opt`" %
             (self.name, source_name, elapsed))


    def _cmd_compile(self, cmd, input=None):
        stdin = subprocess.PIPE if input is not None else None

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

    @property
    def _soname(self):
        return self._tf.name

    def setup_memref_args(self):
        """
        Add memrefs to args dictionary so they can be passed to the cfunction
        """
        args = dict()
        for arg in self.functions:
            if isinstance(arg, TimeFunction):
                data = arg._data_allocated
                # iterate over the first dimension (time)
                for t in range(data.shape[0]):
                    args[f'{arg._C_name}_{t}'] = data[t, ...].ctypes.data_as(ptr_of(f32))
        self._jit_kernel_constants.update(args)

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        return kwargs

    @classmethod
    def _check_kwargs(cls, **kwargs):
        return

    @classmethod
    def _build(cls, expressions, **kwargs) -> Callable:
        debug("-Building operator")
        # Python- (i.e., compile-) and C-level (i.e., run-time) performance
        profiler = create_profile('timers')

        # Lower the input expressions into an IET
        debug("-Lower expressions")
        irs, _, module = cls._lower(expressions, profiler=profiler, **kwargs)

        # Make it an actual Operator
        op = Callable.__new__(cls, **irs.iet.args)
        Callable.__init__(op, **op.args)

        # Header files, etc.
        # op._headers = OrderedSet(*cls._default_headers)
        # op._headers.update(byproduct.headers)
        # op._globals = OrderedSet(*cls._default_globals)
        # op._includes = OrderedSet(*cls._default_includes)
        # op._includes.update(profiler._default_includes)
        # op._includes.update(byproduct.includes)
        op._module = module

        # Required for the jit-compilation
        op._compiler = kwargs['compiler']
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
        # op._func_table.update(OrderedDict([(i.root.name, i) for i in byproduct.funcs]))

        # Internal mutable state to store information about previous runs, autotuning
        # reports, etc
        op._state = cls._initialize_state(**kwargs)

        # Produced by the various compilation passes
        op._reads = filter_sorted(flatten(e.reads for e in irs.expressions))
        op._writes = filter_sorted(flatten(e.writes for e in irs.expressions))
        op._dimensions = set().union(*[e.dimensions for e in irs.expressions])
        op._dtype, op._dspace = irs.clusters.meta
        op._profiler = profiler

        return op

    def __init__(self, *args, **kwargs):
        # Bypass the silent call to __init__ triggered through the backends engine
        pass

    # Compilation -- Expression level

    @classmethod
    def _lower(cls, expressions, **kwargs):
        """
        Perform the lowering Expressions -> Clusters -> ScheduleTree -> IET.
        """
        # Create a symbol registry
        kwargs['sregistry'] = SymbolRegistry()

        expressions = as_tuple(expressions)

        # Input check
        if any(not isinstance(i, Evaluable) for i in expressions):
            raise InvalidOperator("Only `devito.Evaluable` are allowed.")

        # Enable recursive lowering
        # This may be used by a compilation pass that constructs a new
        # expression for which a partial or complete lowering is desired
        kwargs['lower'] = cls._lower

        # [Eq] -> [LoweredEq]
        expressions = cls._lower_exprs(expressions, **kwargs)

        conv = ExtractDevitoStencilConversion(expressions)
        module = conv.convert()
        convert_devito_stencil_to_xdsl_stencil(module, timed=True)

        # [LoweredEq] -> [Clusters]
        clusters = cls._lower_clusters(expressions, **kwargs)

        # [Clusters] -> ScheduleTree
        stree = cls._lower_stree(clusters, **kwargs)

        # ScheduleTree -> unbounded IET
        uiet = cls._lower_uiet(stree, **kwargs)

        # unbounded IET -> IET
        iet, byproduct = cls._lower_iet(uiet, **kwargs)

        return IRs(expressions, clusters, stree, uiet, iet), byproduct, module

    @classmethod
    def _rcompile_wrapper(cls, **kwargs):
        def wrapper(expressions, kwargs=kwargs):
            return rcompile(expressions, kwargs)
        return wrapper

    @classmethod
    def _initialize_state(cls, **kwargs):
        return {}

    @classmethod
    def _specialize_dsl(cls, expressions, **kwargs):
        """
        Backend hook for specialization at the DSL level. The input is made of
        expressions and other higher order objects such as Injection or
        Interpolation; the expressions are still unevaluated at this stage,
        meaning that they are still in tensorial form and derivatives aren't
        expanded yet.
        """
        return expressions

    @classmethod
    def _specialize_exprs(cls, expressions, **kwargs):
        """
        Backend hook for specialization at the expression level.
        """
        return expressions

    @classmethod
    @timed_pass(name='lowering.Expressions')
    def _lower_exprs(cls, expressions, **kwargs):
        """
        Expression lowering:

            * Apply rewrite rules;
            * Evaluate derivatives;
            * Flatten vectorial equations;
            * Indexify Functions;
            * Apply substitution rules;
            * Shift indices for domain alignment.
        """
        expand = kwargs['options'].get('expand', True)

        # Specialization is performed on unevaluated expressions
        expressions = cls._specialize_dsl(expressions, **kwargs)

        # Lower FD derivatives
        # NOTE: we force expansion of derivatives along SteppingDimensions
        # because it drastically simplifies the subsequent lowering into
        # ModuloDimensions
        if not expand:
            expand = lambda d: d.is_Stepping
        expressions = flatten([i._evaluate(expand=expand) for i in expressions])

        # Scalarize the tensor equations, if any
        expressions = [j for i in expressions for j in i._flatten]

        # A second round of specialization is performed on evaluated expressions
        expressions = cls._specialize_exprs(expressions, **kwargs)

        # "True" lowering (indexification, shifting, ...)
        expressions = lower_exprs(expressions, **kwargs)

        processed = [LoweredEq(i) for i in expressions]

        return processed

    # Compilation -- Cluster level

    @classmethod
    def _specialize_clusters(cls, clusters, **kwargs):
        """
        Backend hook for specialization at the Cluster level.
        """
        return clusters

    @classmethod
    @timed_pass(name='lowering.Clusters')
    def _lower_clusters(cls, expressions, profiler=None, **kwargs):
        """
        Clusters lowering:

            * Group expressions into Clusters;
            * Introduce guards for conditional Clusters;
            * Analyze Clusters to detect computational properties such
              as parallelism.
            * Optimize Clusters for performance
        """
        sregistry = kwargs['sregistry']

        # Build a sequence of Clusters from a sequence of Eqs
        clusters = clusterize(expressions, **kwargs)

        # Operation count before specialization
        init_ops = sum(estimate_cost(c.exprs) for c in clusters if c.is_dense)

        clusters = cls._specialize_clusters(clusters, **kwargs)

        # Operation count after specialization
        final_ops = sum(estimate_cost(c.exprs) for c in clusters if c.is_dense)
        try:
            profiler.record_ops_variation(init_ops, final_ops)
        except AttributeError:
            pass

        # Generate implicit Clusters from higher level abstractions
        clusters = generate_implicit(clusters, sregistry=sregistry)

        # Lower all remaining high order symbolic objects
        clusters = lower_index_derivatives(clusters, **kwargs)

        # Make sure no reconstructions can unpick any of the symbolic
        # optimizations performed so far
        clusters = unevaluate(clusters)

        return ClusterGroup(clusters)

    # Compilation -- ScheduleTree level

    @classmethod
    def _specialize_stree(cls, stree, **kwargs):
        """
        DEPRECATED: Backend hook for specialization at the Schedule tree level.
        """
        return stree

    @classmethod
    @timed_pass(name='lowering.ScheduleTree')
    def _lower_stree(cls, clusters, **kwargs):
        """
        Schedule tree lowering:

            * Turn a sequence of Clusters into a ScheduleTree;
            * Derive and attach metadata for distributed-memory parallelism;
            * Derive sections for performance profiling
        """
        # DEPRECATED: Build a ScheduleTree from a sequence of Clusters
        stree = stree_build(clusters, **kwargs)
        stree = cls._specialize_stree(stree)

        return stree

    # Compilation -- Iteration/Expression tree level

    @classmethod
    def _specialize_iet(cls, graph, **kwargs):
        """
        Backend hook for specialization at the Iteration/Expression tree level.
        """
        return graph

    @classmethod
    @timed_pass(name='lowering.uIET')
    def _lower_uiet(cls, stree, profiler=None, **kwargs):
        """
        Turn a ScheduleTree into an unbounded Iteration/Expression tree, that is
        in essence a "floating" IET where one or more variables may be unbounded
        (i.e., no definition placed yet).
        """
        # Build an unbounded IET from a ScheduleTree
        uiet = iet_build(stree)

        # Analyze the IET Sections for C-level profiling
        try:
            profiler.analyze(uiet)
        except AttributeError:
            pass

        return uiet

    @classmethod
    @timed_pass(name='lowering.IET')
    def _lower_iet(cls, uiet, profiler=None, **kwargs):
        """
        Iteration/Expression tree lowering:

            * Introduce distributed-memory, shared-memory, and SIMD parallelism;
            * Introduce optimizations for data locality;
            * Finalize (e.g., symbol definitions, array casts)
        """
        name = kwargs.get("name", "Kernel")
        sregistry = kwargs['sregistry']

        # Wrap the IET with an EntryFunction (a special Callable representing
        # the entry point of the generated library)
        parameters = derive_parameters(uiet, True)
        iet = EntryFunction(name, uiet, 'int', parameters, ())

        # Lower IET to a target-specific IET
        graph = Graph(iet, sregistry=sregistry)
        graph = cls._specialize_iet(graph, **kwargs)

        # Instrument the IET for C-level profiling
        # Note: this is postponed until after _specialize_iet because during
        # specialization further Sections may be introduced
        cls._Target.instrument(graph, profiler=profiler, **kwargs)

        # Extract the necessary macros from the symbolic objects
        generate_macros(graph)

        # Target-independent optimizations
        minimize_symbols(graph)

        return graph.root, graph

    # Read-only properties exposed to the outside world

    @cached_property
    def reads(self):
        return tuple(self._reads)

    @cached_property
    def writes(self):
        return tuple(self._writes)

    @cached_property
    def dimensions(self):
        ret = set().union(*[d._defines for d in self._dimensions])

        # During compilation other Dimensions may have been produced
        dimensions = FindSymbols('dimensions').visit(self)
        ret.update(d for d in dimensions if d.is_PerfKnob)

        ret = tuple(sorted(ret, key=attrgetter('name')))

        return ret

    @cached_property
    def input(self):
        return tuple(i for i in self.parameters if i.is_Input)

    @cached_property
    def temporaries(self):
        return tuple(i for i in self.parameters if i.is_TempFunction)

    @cached_property
    def objects(self):
        return tuple(i for i in self.parameters if i.is_Object)

    # Arguments processing

    @cached_property
    def _access_modes(self):
        """
        A table providing the AccessMode of all user-accessible symbols in `self`.
        """
        return frozendict({i: AccessMode(i in self.reads, i in self.writes)
                           for i in self.input})

    def _prepare_arguments(self, autotune=None, **kwargs):
        """
        Process runtime arguments passed to ``.apply()` and derive
        default values for any remaining arguments.
        """
        # Sanity check -- all user-provided keywords must be known to the Operator
        if not configuration['ignore-unknowns']:
            for k, v in kwargs.items():
                if k not in self._known_arguments:
                    raise ValueError("Unrecognized argument %s=%s" % (k, v))

        # Pre-process Dimension overrides. This may help ruling out ambiguities
        # when processing the `defaults` arguments. A topological sorting is used
        # as DerivedDimensions may depend on their parents
        nodes = self.dimensions
        edges = [(i, i.parent) for i in self.dimensions
                 if i.is_Derived and i.parent in set(nodes)]
        toposort = DAG(nodes, edges).topological_sort()

        futures = {}
        for d in reversed(toposort):
            if set(d._arg_names).intersection(kwargs):
                futures.update(d._arg_values(self._dspace[d], args={}, **kwargs))

        overrides, defaults = split(self.input, lambda p: p.name in kwargs)

        # Process data-carrier overrides
        args = kwargs['args'] = ReducerMap()
        for p in overrides:
            args.update(p._arg_values(**kwargs))
            try:
                args.reduce_inplace()
            except ValueError:
                raise ValueError("Override `%s` is incompatible with overrides `%s`" %
                                 (p, [i for i in overrides if i.name in args]))

        # Process data-carrier defaults
        for p in defaults:
            if p.name in args:
                # E.g., SubFunctions
                continue
            for k, v in p._arg_values(**kwargs).items():
                if k not in args:
                    args[k] = v
                elif k in futures:
                    # An explicit override is later going to set `args[k]`
                    pass
                elif k in kwargs:
                    # User is in control
                    # E.g., given a ConditionalDimension `t_sub` with factor `fact` and
                    # a TimeFunction `usave(t_sub, x, y)`, an override for `fact` is
                    # supplied w/o overriding `usave`; that's legal
                    pass
                elif is_integer(args[k]) and not contains_val(args[k], v):
                    raise ValueError("Default `%s` is incompatible with other args as "
                                     "`%s=%s`, while `%s=%s` is expected. Perhaps you "
                                     "forgot to override `%s`?" %
                                     (p, k, v, k, args[k], p))

        args = kwargs['args'] = args.reduce_all()

        # DiscreteFunctions may be created from CartesianDiscretizations, which in
        # turn could be Grids or SubDomains. Both may provide arguments
        discretizations = {getattr(kwargs[p.name], 'grid', None) for p in overrides}
        discretizations.update({getattr(p, 'grid', None) for p in defaults})
        discretizations.discard(None)
        # Remove subgrids if multiple grids
        if len(discretizations) > 1:
            discretizations = {g for g in discretizations
                               if not any(d.is_Derived for d in g.dimensions)}

        for i in discretizations:
            args.update(i._arg_values(**kwargs))

        # There can only be one Grid from which DiscreteFunctions were created
        grids = {i for i in discretizations if isinstance(i, Grid)}
        if len(grids) > 1:
            # We loosely tolerate multiple Grids for backwards compatibility
            # with spacial subsampling, which should be revisited however. And
            # With MPI it would definitely break!
            if configuration['mpi']:
                raise ValueError("Multiple Grids found")
        try:
            grid = grids.pop()
        except KeyError:
            grid = None

        # An ArgumentsMap carries additional metadata that may be used by
        # the subsequent phases of the arguments processing
        args = kwargs['args'] = ArgumentsMap(args, grid, self)

        # Process Dimensions
        for d in reversed(toposort):
            args.update(d._arg_values(self._dspace[d], grid, **kwargs))

        # Process Objects
        for o in self.objects:
            args.update(o._arg_values(grid=grid, **kwargs))

        # In some "lower-level" Operators implementing a random piece of C, such as
        # one or more calls to third-party library functions, there could still be
        # at this point unprocessed arguments (e.g., scalars)
        kwargs.pop('args')
        args.update({k: v for k, v in kwargs.items() if k not in args})

        # Sanity check
        for p in self.parameters:
            p._arg_check(args, self._dspace[p], am=self._access_modes.get(p))
        for d in self.dimensions:
            if d.is_Derived:
                d._arg_check(args, self._dspace[p])

        # Turn arguments into a format suitable for the generated code
        # E.g., instead of NumPy arrays for Functions, the generated code expects
        # pointers to ctypes.Struct
        for p in self.parameters:
            try:
                args.update(kwargs.get(p.name, p)._arg_finalize(args, alias=p))
            except AttributeError:
                # User-provided floats/ndarray obviously do not have `_arg_finalize`
                args.update(p._arg_finalize(args, alias=p))

        # Execute autotuning and adjust arguments accordingly
        args.update(self._autotune(args, autotune or configuration['autotuning']))

        return args

    def _postprocess_arguments(self, args, **kwargs):
        """Process runtime arguments upon returning from ``.apply()``."""
        for p in self.parameters:
            try:
                subfuncs = (args[getattr(p, s).name] for s in p._sub_functions)
                p._arg_apply(args[p.name], *subfuncs, alias=kwargs.get(p.name))
            except AttributeError:
                p._arg_apply(args[p.name], alias=kwargs.get(p.name))

    @cached_property
    def _known_arguments(self):
        """The arguments that can be passed to ``apply`` when running the Operator."""
        ret = set()
        for i in self.input:
            ret.update(i._arg_names)
            try:
                ret.update(i.grid._arg_names)
            except AttributeError:
                pass
        for d in self.dimensions:
            ret.update(d._arg_names)
        ret.update(p.name for p in self.parameters)
        return frozenset(ret)

    def _autotune(self, args, setup):
        """Auto-tuning to improve runtime performance."""
        return args

    def arguments(self, **kwargs):
        """Arguments to run the Operator."""
        args = self._prepare_arguments(**kwargs)
        # Check all arguments are present
        for p in self.parameters:
            if args.get(p.name) is None:
                raise ValueError("No value found for parameter %s" % p.name)
        return args

    # Code generation and JIT compilation

    #@cached_property
    #def _soname(self):
    #    """A unique name for the shared object resulting from JIT compilation."""
    #    return Signer._digest(self, configuration)

    @cached_property
    def ccode(self):
        try:
            return self._ccode_handler(compiler=self._compiler).visit(self)
        except (AttributeError, TypeError):
            from devito.ir.iet.visitors import CGen
            return CGen(compiler=self._compiler).visit(self)

    @property
    def cfunction(self):
        """The JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:
            self._jit_compile()
            self.setup_memref_args()
            self._lib = self._compiler.load(self._tf.name)
            self._lib.name = self._tf.name

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, "apply_kernel")
            # Associate a C type to each argument for runtime type check
            self._cfunction.argtypes = self._construct_cfunction_args(self._jit_kernel_constants, get_types=True)

        return self._cfunction

    def cinterface(self, force=False):
        """
        Generate two files under the prescribed temporary directory:

            * `X.c` (or `X.cpp`): the code generated for this Operator;
            * `X.h`: an header file representing the interface of `X.c`.

        Where `X=self.name`.

        Parameters
        ----------
        force : bool, optional
            Overwrite any existing files. Defaults to False.
        """
        dest = self._compiler.get_jit_dir()
        name = dest.joinpath(self.name)

        cfile = name.with_suffix(".%s" % self._compiler.src_ext)
        hfile = name.with_suffix('.h')

        # Generate the .c and .h code
        ccode, hcode = CInterface().visit(self)

        for f, code in [(cfile, ccode), (hfile, hcode)]:
            if not force and f.is_file():
                debug("`%s` was not saved in `%s` as it already exists" % (f.name, dest))
            else:
                with open(str(f), 'w') as ff:
                    ff.write(str(code))
                debug("`%s` successfully saved in `%s`" % (f.name, dest))

        return ccode, hcode

    # Execution

    def __call__(self, **kwargs):
        return self.apply(**kwargs)

    def apply(self, **kwargs):
        """
        Execute the Operator.

        With no arguments provided, the Operator runs using the data carried by the
        objects appearing in the input expressions -- these are referred to as the
        "default arguments".

        Optionally, any of the Operator default arguments may be replaced by passing
        suitable key-value arguments. Given ``apply(k=v, ...)``, ``(k, v)`` may be
        used to:

        * replace a Constant. In this case, ``k`` is the name of the Constant,
          ``v`` is either a Constant or a scalar value.

        * replace a Function (SparseFunction). Here, ``k`` is the name of the
          Function, ``v`` is either a Function or a numpy.ndarray.

        * alter the iteration interval along a Dimension. Consider a generic
          Dimension ``d`` iterated over by the Operator.  By default, the Operator
          runs over all iterations within the compact interval ``[d_m, d_M]``,
          where ``d_m`` and ``d_M`` are, respectively, the smallest and largest
          integers not causing out-of-bounds memory accesses (for the Grid
          Dimensions, this typically implies iterating over the entire physical
          domain). So now ``k`` can be either ``d_m`` or ``d_M``, while ``v``
          is an integer value.

        Examples
        --------
        Consider the following Operator

        >>> from devito import Eq, Grid, TimeFunction, Operator
        >>> grid = Grid(shape=(3, 3))
        >>> u = TimeFunction(name='u', grid=grid, save=3)
        >>> op = Operator(Eq(u.forward, u + 1))

        The Operator is run by calling ``apply``

        >>> summary = op.apply()

        The variable ``summary`` contains information about runtime performance.
        As no key-value parameters are specified, the Operator runs with its
        default arguments, namely ``u=u, x_m=0, x_M=2, y_m=0, y_M=2, time_m=0,
        time_M=1``.

        At this point, the same Operator can be used for a completely different
        run, for example

        >>> u2 = TimeFunction(name='u', grid=grid, save=5)
        >>> summary = op.apply(u=u2, x_m=1, y_M=1)

        Now, the Operator will run with a different set of arguments, namely
        ``u=u2, x_m=1, x_M=2, y_m=0, y_M=1, time_m=0, time_M=3``.

        To run an Operator that only uses buffered TimeFunctions, the maximum
        iteration point along the time dimension must be explicitly specified
        (otherwise, the Operator wouldn't know how many iterations to run).

        >>> u3 = TimeFunction(name='u', grid=grid)
        >>> op = Operator(Eq(u3.forward, u3 + 1))
        >>> summary = op.apply(time_M=10)
        """
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

    def _construct_cfunction_args(self, args, get_types = False):
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

    def _emit_build_profiling(self):
        if not is_log_enabled_for('PERF'):
            return

        # Rounder to K decimal places
        fround = lambda i, n=100: ceil(i * n) / n

        timings = self._profiler.py_timers.copy()

        tot = timings.pop('op-compile')
        perf("Operator `%s` generated in %.2f s" % (self.name, fround(tot)))

        max_hotspots = 3
        threshold = 20.

        def _emit_timings(timings, indent=''):
            timings.pop('total', None)
            entries = sorted(timings, key=lambda i: timings[i]['total'], reverse=True)
            for i in entries[:max_hotspots]:
                v = fround(timings[i]['total'])
                perc = fround(v/tot*100, n=10)
                if perc > threshold:
                    perf("%s%s: %.2f s (%.1f %%)" % (indent, i.lstrip('_'), v, perc))
                    _emit_timings(timings[i], ' '*len(indent) + ' * ')

        _emit_timings(timings, '  * ')

        if self._profiler._ops:
            ops = ['%d --> %d' % i for i in self._profiler._ops]
            perf("Flops reduction after symbolic optimization: [%s]" % ' ; '.join(ops))

    def _emit_apply_profiling(self, args):
        """Produce a performance summary of the profiled sections."""
        # Rounder to 2 decimal places
        fround = lambda i: ceil(i * 100) / 100

        elapsed = fround(self._profiler.py_timers['apply'])
        info("Operator `%s` ran in %.2f s" % (self.name, elapsed))

        summary = self._profiler.summary(args, self._dtype, reduce_over=elapsed)

        if not is_log_enabled_for('PERF'):
            # Do not waste time
            return summary

        if summary.globals:
            # Note that with MPI enabled, the global performance indicators
            # represent "cross-rank" performance data
            metrics = []

            v = summary.globals.get('vanilla')
            if v is not None:
                metrics.append("OI=%.2f" % fround(v.oi))
                metrics.append("%.2f GFlops/s" % fround(v.gflopss))

            v = summary.globals.get('fdlike')
            if v is not None:
                metrics.append("%.2f GPts/s" % fround(v.gpointss))

            if metrics:
                perf("Global performance: [%s]" % ', '.join(metrics))

            perf("Local performance:")
            indent = " "*2
        else:
            indent = ""

            if isinstance(self._profiler, AdvancedProfilerVerbose):
                metrics = []

                v = summary.globals.get('fdlike-nosetup')
                if v is not None:
                    metrics.append("%.2f GPts/s" % fround(v.gpointss))

                if metrics:
                    perf("Global performance <w/o setup>: [%s]" % ', '.join(metrics))

        # Emit local, i.e. "per-rank" performance. Without MPI, this is the only
        # thing that will be emitted
        def lower_perfentry(v):
            if v.gflopss:
                oi = "OI=%.2f" % fround(v.oi)
                gflopss = "%.2f GFlops/s" % fround(v.gflopss)
                gpointss = "%.2f GPts/s" % fround(v.gpointss)
                return "[%s]" % ", ".join([oi, gflopss, gpointss])
            elif v.gpointss:
                gpointss = "%.2f GPts/s" % fround(v.gpointss)
                return "[%s]" % gpointss
            else:
                return ""

        for k, v in summary.items():
            rank = "[rank%d]" % k.rank if k.rank is not None else ""

            metrics = lower_perfentry(v)

            itershapes = [",".join(str(i) for i in its) for its in v.itershapes]
            if len(itershapes) > 1:
                itershapes = ",".join("<%s>" % i for i in itershapes)
            elif len(itershapes) == 1:
                itershapes = itershapes[0]
            else:
                itershapes = ""
            name = "%s%s<%s>" % (k.name, rank, itershapes)

            perf("%s* %s ran in %.2f s %s" % (indent, name, fround(v.time), metrics))
            for n, v1 in summary.subsections.get(k.name, {}).items():
                metrics = lower_perfentry(v1)

                perf("%s+ %s ran in %.2f s [%.2f%%] %s" %
                     (indent*2, n, fround(v1.time), fround(v1.time/v.time*100),
                      metrics))

        # Emit performance mode and arguments
        perf_args = {}
        for i in self.input + self.dimensions:
            if not i.is_PerfKnob:
                continue
            try:
                perf_args[i.name] = args[i.name]
            except KeyError:
                # Try with the aliases
                for a in i._arg_names:
                    if a in args:
                        perf_args[a] = args[a]
                        break
        perf("Performance[mode=%s] arguments: %s" % (self._mode, perf_args))

        return summary

    # Pickling support

    def __getstate__(self):
        if self._lib:
            state = dict(self.__dict__)
            # The compiled shared-object will be pickled; upon unpickling, it
            # will be restored into a potentially different temporary directory,
            # so the entire process during which the shared-object is loaded and
            # given to ctypes must be performed again
            state['_lib'] = None
            state['_cfunction'] = None
            # Do not pickle the `args` used to construct the Operator. Not only
            # would this be completely useless, but it might also lead to
            # allocating additional memory upon unpickling, as the user-provided
            # equations typically carry different instances of the same Function
            # (e.g., f(t, x-1), f(t, x), f(t, x+1)), which are different objects
            # with distinct `.data` fields
            state['_args'] = None
            with open(self._lib._name, 'rb') as f:
                state['binary'] = f.read()
                state['soname'] = self._soname
            return state
        else:
            return self.__dict__

    def __getnewargs_ex__(self):
        return (None,), {}

    def __setstate__(self, state):
        soname = state.pop('soname', None)
        binary = state.pop('binary', None)
        for k, v in state.items():
            setattr(self, k, v)
        if soname is not None:
            self._compiler.save(soname, binary)
            self._lib = self._compiler.load(soname)
            self._lib.name = soname


# Default action (perform or bypass) for selected compilation passes upon
# recursive compilation
# NOTE: it may not only be pointless to apply the following passes recursively
# (because once, during the main compilation phase, is simply enough), but also
# dangerous as some of them (the minority) might break in some circumstances
# if applied in cascade (e.g., `linearization` on top of `linearization`)
rcompile_registry = {
    'mpi': False,
    'linearize': False,
    'place-transfers': False
}


def rcompile(expressions, kwargs=None):
    """
    Perform recursive compilation on an ordered sequence of symbolic expressions.
    """
    if not kwargs or 'options' not in kwargs:
        kwargs = parse_kwargs(**kwargs)
        cls = operator_selector(**kwargs)
        kwargs = cls._normalize_kwargs(**kwargs)
    else:
        cls = operator_selector(**kwargs)

    # Tweak the compilation kwargs
    options = dict(kwargs['options'])
    options.update(rcompile_registry)
    kwargs['options'] = options

    # Recursive profiling not supported -- would be a complete mess
    kwargs.pop('profiler', None)

    return cls._lower(expressions, **kwargs)


# Misc helpers


IRs = namedtuple('IRs', 'expressions clusters stree uiet iet')


class ArgumentsMap(dict):

    def __init__(self, args, grid, op):
        super().__init__(args)

        self.grid = grid

        self.allocator = op._allocator
        self.platform = op._platform
        # self.language = op._language
        self.compiler = op._compiler
        self.options = op._options

    @property
    def comm(self):
        """The MPI communicator the arguments are collective over."""
        return self.grid.comm if self.grid is not None else MPI.COMM_NULL

    @property
    def opkwargs(self):
        temp_registry = {v: k for k, v in platform_registry.items()}
        platform = temp_registry[self.platform]

        temp_registry = {v: k for k, v in compiler_registry.items()}
        compiler = temp_registry[self.compiler.__class__]

        return {'platform': platform, 'compiler': compiler, 'language': self.language}


def parse_kwargs(**kwargs):
    """
    Parse keyword arguments provided to an Operator.
    """
    # `dse` -- deprecated, dropped
    dse = kwargs.pop("dse", None)
    if dse is not None:
        warning("The `dse` argument is deprecated. "
                "The optimization level is now controlled via the `opt` argument")

    # `dle` -- deprecated, replaced by `opt`
    if 'dle' in kwargs:
        warning("The `dle` argument is deprecated. "
                "The optimization level is now controlled via the `opt` argument")
        dle = kwargs.pop('dle')
        if 'opt' in kwargs:
            warning("Both `dle` and `opt` were passed; ignoring `dle` argument")
            opt = kwargs.pop('opt')
        else:
            warning("Setting `opt=%s`" % str(dle))
            opt = dle
    elif 'opt' in kwargs:
        opt = kwargs.pop('opt')
    else:
        opt = configuration['opt']

    if not opt or isinstance(opt, str):
        mode, options = opt, {}
    elif isinstance(opt, tuple):
        if len(opt) == 0:
            mode, options = 'noop', {}
        elif isinstance(opt[-1], dict):
            if len(opt) == 2:
                mode, options = opt
            else:
                mode, options = tuple(flatten(i.split(',') for i in opt[:-1])), opt[-1]
        else:
            mode, options = tuple(flatten(i.split(',') for i in opt)), {}
    else:
        raise InvalidOperator("Illegal `opt=%s`" % str(opt))

    # `opt`, deprecated kwargs
    kwopenmp = kwargs.get('openmp', options.get('openmp'))
    if kwopenmp is None:
        openmp = kwargs.get('language', configuration['language']) == 'openmp'
    else:
        openmp = kwopenmp

    # `opt`, options
    options = dict(options)
    options.setdefault('openmp', openmp)
    options.setdefault('mpi', configuration['mpi'])
    for k, v in configuration['opt-options'].items():
        options.setdefault(k, v)
    # Handle deprecations
    deprecated_options = ('cire-mincost-inv', 'cire-mincost-sops', 'cire-maxalias')
    for i in deprecated_options:
        try:
            options.pop(i)
            warning("Ignoring deprecated optimization option `%s`" % i)
        except KeyError:
            pass
    kwargs['options'] = options

    # `opt`, mode
    if mode is None:
        mode = 'noop'
    kwargs['mode'] = mode

    # `platform`
    platform = kwargs.get('platform')
    if platform is not None:
        if not isinstance(platform, str):
            raise ValueError("Argument `platform` should be a `str`")
        if platform not in configuration._accepted['platform']:
            raise InvalidOperator("Illegal `platform=%s`" % str(platform))
        kwargs['platform'] = platform_registry[platform]()
    else:
        kwargs['platform'] = configuration['platform']

    # `language`
    language = kwargs.get('language')
    if language is not None:
        if not isinstance(language, str):
            raise ValueError("Argument `language` should be a `str`")
        if language not in configuration._accepted['language']:
            raise InvalidOperator("Illegal `language=%s`" % str(language))
        kwargs['language'] = language
    elif kwopenmp is not None:
        # Handle deprecated `openmp` kwarg for backward compatibility
        kwargs['language'] = 'openmp' if openmp else 'C'
    else:
        kwargs['language'] = configuration['language']

    # `compiler`
    compiler = kwargs.get('compiler')
    if compiler is not None:
        if not isinstance(compiler, str):
            raise ValueError("Argument `compiler` should be a `str`")
        if compiler not in configuration._accepted['compiler']:
            raise InvalidOperator("Illegal `compiler=%s`" % str(compiler))
        kwargs['compiler'] = compiler_registry[compiler](platform=kwargs['platform'],
                                                         language=kwargs['language'],
                                                         mpi=configuration['mpi'])
    elif any([platform, language]):
        kwargs['compiler'] =\
            configuration['compiler'].__new_with__(platform=kwargs['platform'],
                                                   language=kwargs['language'],
                                                   mpi=configuration['mpi'])
    else:
        kwargs['compiler'] = configuration['compiler'].__new_with__()

    # `allocator`
    kwargs['allocator'] = default_allocator(
        '%s.%s.%s' % (kwargs['compiler'].name,
                      kwargs['language'],
                      kwargs['platform'])
    )

    return kwargs


def get_arg_names_from_module(op):
    return [
        str_attr.data 
        for str_attr in op.body.block.ops.first.attributes['param_names'].data
    ]

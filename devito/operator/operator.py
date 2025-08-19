from collections import OrderedDict, namedtuple
from functools import cached_property
import ctypes
import shutil
from operator import attrgetter
from math import ceil
from tempfile import gettempdir

from sympy import sympify
import sympy
import numpy as np

from devito.arch import ANYCPU, Device, compiler_registry, platform_registry
from devito.data import default_allocator
from devito.exceptions import (CompilationError, ExecutionError, InvalidArgument,
                               InvalidOperator)
from devito.logger import (debug, info, perf, warning, is_log_enabled_for,
                           switch_log_level)
from devito.ir.equations import LoweredEq, lower_exprs, concretize_subdims
from devito.ir.clusters import ClusterGroup, clusterize
from devito.ir.iet import (Callable, CInterface, EntryFunction, DeviceFunction,
                           FindSymbols, MetaCall, derive_parameters, iet_build)
from devito.ir.support import AccessMode, SymbolRegistry
from devito.ir.stree import stree_build
from devito.operator.profiling import create_profile
from devito.operator.registry import operator_selector
from devito.mpi import MPI
from devito.parameters import configuration
from devito.passes import (
    Graph, lower_index_derivatives, generate_implicit, generate_macros,
    minimize_symbols, unevaluate, error_mapper, is_on_device, lower_dtypes
)
from devito.symbolics import estimate_cost, subs_op_args
from devito.tools import (DAG, OrderedSet, Signer, ReducerMap, as_mapper, as_tuple,
                          flatten, filter_sorted, frozendict, is_integer,
                          split, timed_pass, timed_region, contains_val,
                          CacheInstances, MemoryEstimate)
from devito.types import (Buffer, Evaluable, host_layer, device_layer,
                          disk_layer)
from devito.types.dimension import Thickness


__all__ = ['Operator']


_layers = (disk_layer, host_layer, device_layer)


class Operator(Callable):

    """
    Generate, JIT-compile and run C code starting from an ordered sequence
    of symbolic expressions.

    Parameters
    ----------
    expressions : expr-like or list or expr-like
        The (list of) expression(s) defining the Operator computation.
    **kwargs
        * name : str
            Name of the Operator, defaults to "Kernel".
        * subs : dict
            Symbolic substitutions to be applied to ``expressions``.
        * opt : str
            The performance optimization level. Defaults to ``configuration['opt']``.
        * language : str
            The target language for shared-memory parallelism. Defaults to
            ``configuration['language']``.
        * platform : str
            The architecture the code is generated for. Defaults to
            ``configuration['platform']``.
        * compiler : str
            The backend compiler used to jit-compile the generated code.
            Defaults to ``configuration['compiler']``.

    Examples
    --------
    The following Operator implements a trivial time-marching method that
    adds 1 to every grid point in ``u`` at every timestep.

    >>> from devito import Eq, Grid, TimeFunction, Operator
    >>> grid = Grid(shape=(4, 4))
    >>> u = TimeFunction(name='u', grid=grid)
    >>> op = Operator(Eq(u.forward, u + 1))

    Multiple expressions can be supplied, and there is no limit to the number of
    expressions in an Operator.

    >>> v = TimeFunction(name='v', grid=grid)
    >>> op = Operator([Eq(u.forward, u + 1),
    ...                Eq(v.forward, v + 1)])

    Simple boundary conditions can be imposed easily exploiting the "indexed
    notation" for Functions/TimeFunctions.

    >>> t = grid.stepping_dim
    >>> x, y = grid.dimensions
    >>> op = Operator([Eq(u.forward, u + 1),
    ...                Eq(u[t+1, x, 0], 0),
    ...                Eq(u[t+1, x, 2], 0),
    ...                Eq(u[t+1, 0, y], 0),
    ...                Eq(u[t+1, 2, y], 0)])

    A semantically equivalent computation can be expressed exploiting SubDomains.
    This is the only way for expressing BCs, when running with MPI.

    >>> u.data[:] = 0
    >>> op = Operator(Eq(u.forward, u + 1, subdomain=grid.interior))

    By specifying a SubDomain, the Operator constrains the execution of an expression to
    a certain sub-region within the computational domain. Ad-hoc SubDomains can also be
    created in application code -- refer to the SubDomain documentation for more info.

    Advanced boundary conditions can be expressed leveraging `SubDomain` and
    `SubDimension`.

    Tensor contractions are supported, but with one caveat: in case of MPI execution, any
    global reductions along an MPI-distributed Dimension should be handled explicitly in
    user code. The following example shows how to implement the matrix-vector
    multiplication ``Av = b`` (inducing a reduction along ``y``).

    >>> from devito import Inc, Function
    >>> A = Function(name='A', grid=grid)
    >>> v = Function(name='v', shape=(3,), dimensions=(y,))
    >>> b = Function(name='b', shape=(3,), dimensions=(x,))
    >>> op = Operator(Inc(b, A*v))

    Dense and sparse computation may be present within the same Operator. In the
    following example, interpolation is used to approximate the value of four
    sparse points placed at the center of the four quadrants at the grid corners.

    >>> import numpy as np
    >>> from devito import SparseFunction
    >>> grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
    >>> f = Function(name='f', grid=grid)
    >>> coordinates = np.array([(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)])
    >>> sf = SparseFunction(name='sf', grid=grid, npoint=4, coordinates=coordinates)
    >>> op = Operator([Eq(f, f + 1)] + sf.interpolate(f))

    The iteration direction is automatically detected by the Devito compiler. Below,
    the Operator runs from ``time_M`` (maximum point in the time dimension) down to
    ``time_m`` (minimum point in the time dimension), as opposed to all of the examples
    seen so far, in which the execution along time proceeds from ``time_m`` to ``time_M``
    through unit-step increments.

    >>> op = Operator(Eq(u.backward, u + 1))

    Loop-level optimisations, including SIMD vectorisation and OpenMP parallelism, are
    automatically discovered and handled by the Devito compiler. For more information,
    refer to the relevant documentation.
    """

    def __new__(cls, expressions, **kwargs):
        if expressions is None:
            # Return a dummy Callable. This is exploited by unpickling. Users
            # can't do anything useful with it
            return super().__new__(cls, **kwargs)

        # Parse input arguments
        kwargs = parse_kwargs(**kwargs)

        # The Operator type for the given target
        cls = operator_selector(**kwargs)

        # Preprocess input arguments
        kwargs = cls._normalize_kwargs(**kwargs)
        cls._check_kwargs(**kwargs)
        expressions = cls._sanitize_exprs(expressions, **kwargs)

        # Lower to a JIT-compilable object
        with timed_region('op-compile') as r:
            op = cls._build(expressions, **kwargs)
        op._profiler.py_timers.update(r.timings)

        # Emit info about how long it took to perform the lowering
        op._emit_build_profiling()

        return op

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        return kwargs

    @classmethod
    def _check_kwargs(cls, **kwargs):
        return

    @classmethod
    def _sanitize_exprs(cls, expressions, **kwargs):
        expressions = as_tuple(expressions)

        for i in expressions:
            if not isinstance(i, Evaluable):
                raise CompilationError(f"`{i!s}` is not an Evaluable object; "
                                       "check your equation again")

        return expressions

    @classmethod
    def _build(cls, expressions, **kwargs):
        # Python- (i.e., compile-) and C-level (i.e., run-time) performance
        profiler = create_profile('timers')

        # Lower the input expressions into an IET
        irs, byproduct = cls._lower(expressions, profiler=profiler, **kwargs)

        # Make it an actual Operator
        op = Callable.__new__(cls, **irs.iet.args)
        Callable.__init__(op, **op.args)

        # Header files, etc.
        op._headers = OrderedSet(*byproduct.headers)
        op._globals = OrderedSet(*byproduct.globals)
        op._includes = OrderedSet(*profiler._default_includes)
        op._includes.update(byproduct.includes)
        op._namespaces = OrderedSet(*byproduct.namespaces)

        # Required for the jit-compilation
        op._compiler = kwargs['compiler']

        # Required for compilation by the profiler
        op._compiler.add_include_dirs(profiler._include_dirs)
        op._compiler.add_library_dirs(profiler._lib_dirs, rpath=True)
        op._compiler.add_libraries(profiler._default_libs)

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
        op._func_table.update(OrderedDict([(i.root.name, i)
                                           for i in byproduct.funcs]))

        # Internal mutable state to store information about previous runs,
        # autotuning reports, etc
        op._state = cls._initialize_state(**kwargs)

        # Produced by the various compilation passes
        op._reads = filter_sorted(flatten(e.reads for e in irs.expressions))
        op._writes = filter_sorted(flatten(e.writes for e in irs.expressions))
        op._dimensions = set().union(*[e.dimensions for e in irs.expressions])
        op._dtype, op._dspace = irs.clusters.meta
        op._profiler = profiler

        # Clear build-scoped instance caches
        CacheInstances.clear_caches()

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
        kwargs.setdefault('sregistry', SymbolRegistry())
        # Add lang-base kwargs
        kwargs.setdefault('langbb', cls._Target.langbb())
        kwargs.setdefault('printer', cls._Target.Printer)

        expressions = as_tuple(expressions)

        # Enable recursive lowering
        # This may be used by a compilation pass that constructs a new
        # expression for which a partial or complete lowering is desired
        kwargs['rcompile'] = cls._rcompile_wrapper(**kwargs)

        # [Eq] -> [LoweredEq]
        expressions = cls._lower_exprs(expressions, **kwargs)

        # [LoweredEq] -> [Clusters]
        clusters = cls._lower_clusters(expressions, **kwargs)

        # [Clusters] -> ScheduleTree
        stree = cls._lower_stree(clusters, **kwargs)

        # ScheduleTree -> unbounded IET
        uiet = cls._lower_uiet(stree, **kwargs)

        # unbounded IET -> IET
        iet, byproduct = cls._lower_iet(uiet, **kwargs)

        return IRs(expressions, clusters, stree, uiet, iet), byproduct

    @classmethod
    def _rcompile_wrapper(cls, **kwargs0):
        raise NotImplementedError

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

        # Turn user-defined SubDimensions into concrete SubDimensions,
        # in particular uniqueness across expressions is ensured
        expressions = concretize_subdims(expressions, **kwargs)

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
        clusters = generate_implicit(clusters)

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
        Backend hook for specialization at the Schedule tree level.
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
        # Build a ScheduleTree from a sequence of Clusters
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

        # Wrap the IET with an EntryFunction (a special Callable representing
        # the entry point of the generated library)
        parameters = derive_parameters(uiet, True)
        iet = EntryFunction(name, uiet, 'int', parameters, ())

        # Lower IET to a target-specific IET
        graph = Graph(iet, **kwargs)
        graph = cls._specialize_iet(graph, **kwargs)

        # Instrument the IET for C-level profiling
        # Note: this is postponed until after _specialize_iet because during
        # specialization further Sections may be introduced
        cls._Target.instrument(graph, profiler=profiler, **kwargs)

        # Extract the necessary macros from the symbolic objects
        generate_macros(graph, **kwargs)

        # Target-specific lowering
        lower_dtypes(graph, **kwargs)

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
    def transients(self):
        return tuple(i for i in self.parameters
                     if i.is_AbstractFunction and i.is_transient)

    @cached_property
    def objects(self):
        return tuple(i for i in self.parameters if i.is_Object)

    @cached_property
    def threads_info(self):
        return frozendict({'nthreads': self.nthreads, 'npthreads': self.npthreads})

    # Arguments processing

    @cached_property
    def _access_modes(self):
        """
        A table providing the AccessMode of all user-accessible symbols in `self`.
        """
        return frozendict({i: AccessMode(i in self.reads, i in self.writes)
                           for i in self.input})

    def _prepare_arguments(self, autotune=None, estimate_memory=False, **kwargs):
        """
        Process runtime arguments passed to ``.apply()` and derive
        default values for any remaining arguments.
        """
        # Sanity check -- all user-provided keywords must be known to the Operator
        if not configuration['ignore-unknowns']:
            for k, v in kwargs.items():
                if k not in self._known_arguments:
                    raise InvalidArgument(f"Unrecognized argument `{k}={v}`")

        overrides, defaults = split(self.input, lambda p: p.name in kwargs)

        # DiscreteFunctions may be created from CartesianDiscretizations, which in
        # turn could be Grids or SubDomains. Both may provide arguments
        discretizations = {getattr(kwargs.get(p.name, p), 'grid', None)
                           for p in self.input} - {None}

        # There can only be one Grid from which DiscreteFunctions were created
        grids = {i.root for i in discretizations}

        if len(grids) > 1 and configuration['mpi']:
            # We loosely tolerate multiple Grids for backwards compatibility
            # with spatial subsampling, which should be revisited however. And
            # With MPI it would definitely break!
            raise ValueError("Multiple Grids found")

        nodes = set(self.dimensions)
        if grids:
            grid = grids.pop()
            nodes.update(grid.dimensions)
        else:
            grid = None

        # Pre-process Dimension overrides. This may help ruling out ambiguities
        # when processing the `defaults` arguments. A topological sorting is used
        # as DerivedDimensions may depend on their parents
        edges = [(i, i.parent) for i in nodes
                 if i.is_Derived and i.parent in nodes]
        toposort = DAG(nodes, edges).topological_sort()

        futures = {}
        for d in reversed(toposort):
            if set(d._arg_names).intersection(kwargs):
                futures.update(d._arg_values(self._dspace[d], args={}, **kwargs))

        # Prepare to process data-carriers
        args = kwargs['args'] = ReducerMap()

        kwargs['metadata'] = {'language': self._language,
                              'platform': self._platform,
                              'transients': self.transients,
                              **self.threads_info}

        overrides, defaults = split(self.input, lambda p: p.name in kwargs)

        # Process data-carrier overrides
        for p in overrides:
            args.update(p._arg_values(estimate_memory=estimate_memory, **kwargs))
            try:
                args.reduce_inplace()
            except ValueError:
                v = [i for i in overrides if i.name in args]
                raise InvalidArgument(
                    f"Override `{p}` is incompatible with overrides `{v}`"
                )

        # Process data-carrier defaults
        for p in defaults:
            if p.name in args:
                # E.g., SubFunctions
                continue
            for k, v in p._arg_values(estimate_memory=estimate_memory, **kwargs).items():
                if k not in args:
                    args[k] = v
                elif k in futures:
                    # An explicit override is later going to set `args[k]`
                    pass
                elif k in kwargs:
                    # User is in control
                    # E.g., given a ConditionalDimension `t_sub` with factor `fact`
                    # and a TimeFunction `usave(t_sub, x, y)`, an override for
                    # `fact` is supplied w/o overriding `usave`; that's legal
                    pass
                elif is_integer(args[k]) and not contains_val(args[k], v):
                    raise InvalidArgument(
                        f"Default `{p}` is incompatible with other args as "
                        f"`{k}={v}`, while `{k}={args[k]}` is expected. Perhaps "
                        f"you forgot to override `{p}`?"
                    )

        args = kwargs['args'] = args.reduce_all()

        for i in discretizations:
            args.update(i._arg_values(**kwargs))

        # An ArgumentsMap carries additional metadata that may be used by
        # the subsequent phases of the arguments processing
        args = kwargs['args'] = ArgumentsMap(args, grid, self)

        if estimate_memory:
            # No need to do anything more if only checking the memory
            return args

        # Process Dimensions
        for d in reversed(toposort):
            args.update(d._arg_values(self._dspace[d], grid, **kwargs))

        # Process Thicknesses
        for p in self.parameters:
            if isinstance(p, Thickness):
                args.update(p._arg_values(grid=grid, **kwargs))

        # Process Objects
        for o in self.objects:
            args.update(o._arg_values(grid=grid, **kwargs))

        # Purge `kwargs`
        kwargs.pop('args')
        kwargs.pop('metadata')

        # In some "lower-level" Operators implementing a random piece of C, such as
        # one or more calls to third-party library functions, there could still be
        # at this point unprocessed arguments (e.g., scalars)
        args.update({k: v for k, v in kwargs.items() if k not in args})

        # Sanity check
        for p in self.parameters:
            p._arg_check(args, self._dspace[p], am=self._access_modes.get(p),
                         **kwargs)
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

    def _postprocess_errors(self, retval):
        if retval == 0:
            return
        elif retval == error_mapper['Stability']:
            raise ExecutionError("Detected nan/inf in some output Functions")
        elif retval == error_mapper['KernelLaunch']:
            raise ExecutionError("Kernel launch failed")
        elif retval == error_mapper['KernelLaunchOutOfResources']:
            raise ExecutionError(
                "Kernel launch failed due to insufficient resources. This may be "
                "due to excessive register pressure in one of the Operator "
                "kernels. Try supplying a smaller `par-tile` value."
            )
        elif retval == error_mapper['KernelLaunchClusterConfig']:
            raise ExecutionError(
                "Kernel launch failed due to an invalid thread block cluster "
                "configuration. This is probably due to a `tbc-tile` value that "
                "does not perfectly divide the number of blocks launched for a "
                "kernel. This is a known, strong limitation which effectively "
                "prevents the use of `tbc-tile` in realistic scenarios, but it "
                "will be removed in future versions."
            )
        elif retval == error_mapper['KernelLaunchUnknown']:
            raise ExecutionError(
                "Kernel launch failed due to an unknown error. This might "
                "simply indicate memory corruption, but also, in a more unlikely "
                "case, a hardware issue. Please report this issue to the "
                "Devito team.")
        else:
            raise ExecutionError("An error occurred during execution")

    def _postprocess_arguments(self, args, **kwargs):
        """Process runtime arguments upon returning from ``.apply()``."""
        for p in self.parameters:
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
                raise InvalidArgument(f"No value found for parameter {p.name}")
        return args

    # Code generation and JIT compilation

    @cached_property
    def _soname(self):
        """A unique name for the shared object resulting from JIT compilation."""
        return Signer._digest(self, configuration)

    @cached_property
    def _printer(self):
        return self._Target.Printer

    @cached_property
    def description(self):
        return f"Devito generated code for Operator `{self.name}`"

    @cached_property
    def headers(self):
        return OrderedSet(*self._printer._headers).union(self._headers)

    @cached_property
    def includes(self):
        return OrderedSet(*self._printer._includes).union(self._includes)

    @cached_property
    def namespaces(self):
        return OrderedSet(*self._printer._namespaces).union(self._namespaces)

    @cached_property
    def ccode(self):
        from devito.ir.iet.visitors import CGen
        return CGen(printer=self._printer).visit(self)

    def _jit_compile(self):
        """
        JIT-compile the C code generated by the Operator.

        It is ensured that JIT compilation will only be performed once per
        Operator, reagardless of how many times this method is invoked.
        """
        if self._lib is None:
            with self._profiler.timer_on('jit-compile'):
                recompiled, src_file = self._compiler.jit_compile(self._soname, str(self))

            elapsed = self._profiler.py_timers['jit-compile']
            if recompiled:
                perf(f"Operator `{self.name}` jit-compiled `{src_file}` in "
                     f"{elapsed:.2f} s with `{self._compiler}`")
            else:
                perf(f"Operator `{self.name}` fetched `{src_file}` in "
                     f"{elapsed:.2f} s from jit-cache")

    @property
    def cfunction(self):
        """The JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:
            self._jit_compile()
            self._lib = self._compiler.load(self._soname)
            self._lib.name = self._soname

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            # Associate a C type to each argument for runtime type check
            self._cfunction.argtypes = [i._C_ctype for i in self.parameters]

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

        cfile = name.with_suffix(f".{self._compiler.src_ext}")
        hfile = name.with_suffix('.h')

        # Generate the .c and .h code
        ccode, hcode = CInterface().visit(self)

        for f, code in [(cfile, ccode), (hfile, hcode)]:
            if not force and f.is_file():
                debug(f"`{f.name}` was not saved in `{dest}` as it already exists")
            else:
                with open(str(f), 'w') as ff:
                    ff.write(str(code))
                debug(f"`{f.name}` successfully saved in `{dest}`")

        return ccode, hcode

    # Execution

    def __call__(self, **kwargs):
        return self.apply(**kwargs)

    def estimate_memory(self, **kwargs):
        """
        Estimate the memory consumed by the Operator without touching or allocating any
        data. This interface is designed to mimic `Operator.apply(**kwargs)` and can be
        called with the kwargs for a prospective Operator execution. With no arguments,
        it will simply estimate memory for the default Operator parameters. However, if
        desired, overrides can be supplied (as per `apply`) and these will be used for
        the memory estimate.

        If estimating memory for an Operator which is expected to allocate large arrays,
        it is strongly recommended that one avoids touching the data in Python (thus
        avoiding allocation). `AbstractFunction` types have their data allocated lazily -
        the underlying array is only created at the point at which the `data`,
        `data_with_halo`, etc, attributes are first accessed. Thus by avoiding accessing
        such attributes in the memory estimation script, one can check the nominal memory
        usage of proposed Operators far larger than will fit in system DRAM.

        Note that this estimate will build the Operator in order to factor in memory
        allocation for array temporaries and buffers generated during compilation.

        Parameters
        ----------
        **kwargs: dict
            As per `Operator.apply()`.

        Returns
        -------
        summary: MemoryEstimate
            An estimate of memory consumed in each of the specified locations.
        """
        # Build the arguments list for which to get the memory consumption
        # This is so that the estimate will factor in overrides
        args = self._prepare_arguments(estimate_memory=True, **kwargs)
        mem = args.nbytes_consumed

        memreport = {'host': mem[host_layer], 'device': mem[device_layer]}

        # Extra information for enriched Operators
        extras = self._enrich_memreport(args)
        memreport.update(extras)

        return MemoryEstimate(memreport, name=self.name)

    def _enrich_memreport(self, args):
        # Hook for enriching memory report with additional metadata
        return {}

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
        # Compile the operator before building the arguments list
        # to avoid out of memory with greedy compilers
        cfunction = self.cfunction

        # Build the arguments list to invoke the kernel function
        with self._profiler.timer_on('arguments'):
            args = self.arguments(**kwargs)

        # Invoke kernel function with args
        arg_values = [args[p.name] for p in self.parameters]
        try:
            with self._profiler.timer_on('apply', comm=args.comm):
                retval = cfunction(*arg_values)
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

        # Perform error checking
        self._postprocess_errors(retval)

        # Post-process runtime arguments
        self._postprocess_arguments(args, **kwargs)

        # In case MPI is used restrict result logging to one rank only
        with switch_log_level(comm=args.comm):
            return self._emit_apply_profiling(args)

    # Performance profiling

    def _emit_build_profiling(self):
        if not is_log_enabled_for('PERF'):
            return

        # Rounder to K decimal places
        fround = lambda i, n=100: ceil(i * n) / n

        timings = self._profiler.py_timers.copy()

        tot = timings.pop('op-compile')
        perf(f"Operator `{self.name}` generated in {fround(tot):.2f} s")

        max_hotspots = 3
        threshold = 20.

        def _emit_timings(timings, indent=''):
            timings.pop('total', None)
            entries = sorted(timings, key=lambda i: timings[i]['total'], reverse=True)
            for i in entries[:max_hotspots]:
                v = fround(timings[i]['total'])
                perc = fround(v/tot*100, n=10)
                if perc > threshold:
                    perf(f"{indent}{i.lstrip('_')}: {v:.2f} s ({perc:.1f} %)")
                    _emit_timings(timings[i], ' '*len(indent) + ' * ')

        _emit_timings(timings, '  * ')

        if self._profiler._ops:
            ops = ['%d --> %d' % i for i in self._profiler._ops]
            perf(f"Flops reduction after symbolic optimization: [{' ; '.join(ops)}]")

    def _emit_apply_profiling(self, args):
        """Produce a performance summary of the profiled sections."""
        # Rounder to 2 decimal places
        fround = lambda i: ceil(i * 100) / 100

        elapsed = fround(self._profiler.py_timers['apply'])
        info(f"Operator `{self.name}` ran in {elapsed:.2f} s")

        summary = self._profiler.summary(args, self._dtype, reduce_over=elapsed)

        if not is_log_enabled_for('PERF'):
            # Do not waste time
            return summary

        if summary.globals:
            # NOTE: with MPI enabled, the global performance indicators
            # represent "cross-rank" performance data

            # Print out global performance indicators
            metrics = []

            v = summary.globals.get('vanilla')
            if v is not None:
                if v.oi is not None:
                    metrics.append(f"OI={fround(v.oi):.2f}")
                if v.gflopss is not None and np.isfinite(v.gflopss):
                    metrics.append(f"{fround(v.gflopss):.2f} GFlops/s")

            v = summary.globals.get('fdlike')
            if v is not None:
                metrics.append(f"{fround(v.gpointss):.2f} GPts/s")

            if metrics:
                perf(f"Global performance: [{', '.join(metrics)}]")

            # Same as above, but excluding the setup phase, e.g. the CPU-GPU
            # data transfers in the case of a GPU run, mallocs, frees, etc.
            metrics = []

            v = summary.globals.get('fdlike-nosetup')
            if v is not None:
                metrics.append(f"{fround(v.time):.2f} s")
                metrics.append(f"{fround(v.gpointss):.2f} GPts/s")

                perf(f"Global performance <w/o setup>: [{', '.join(metrics)}]")

            # Prepare for the local performance indicators
            perf("Local performance:")
            indent = " "*2
        else:
            indent = ""

        # Emit local, i.e. "per-rank" performance. Without MPI, this is the only
        # thing that will be emitted
        def lower_perfentry(v):
            values = []
            if v.oi:
                values.append(f"OI={fround(v.oi):.2f}")
            if v.gflopss:
                values.append(f"{fround(v.gflopss):.2f} GFlops/s")
            if v.gpointss:
                values.append(f"{fround(v.gpointss):.2f} GPts/s")

            if values:
                return f"[{', '.join(values)}]"
            else:
                return ""

        for k, v in summary.items():
            rank = f"[rank{k.rank}]" if k.rank is not None else ''
            name = f"{k.name}{rank}"

            if v.time <= 0.01:
                # Trim down the output for very fast sections
                perf(f"{indent}* {name} ran in {fround(v.time):.2f} s")
                continue

            metrics = lower_perfentry(v)
            perf(f"{indent}* {name} ran in {fround(v.time):.2f} s {metrics}")
            for n, v1 in summary.subsections.get(k.name, {}).items():
                metrics = lower_perfentry(v1)

                perf(f"{indent*2}+ {n} ran in {fround(v1.time):.2f} s "
                     f"[{fround(v1.time/v.time*100):.2f}%] {metrics}")

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
        if is_integer(self.npthreads):
            perf_args['pthreads'] = self.npthreads
        perf_args = {k: perf_args[k] for k in sorted(perf_args)}
        perf(f"Performance[mode={self._mode}] arguments: {perf_args}")

        return summary

    # Pickling support

    def __getstate__(self):
        state = dict(self.__dict__)

        if self._lib:
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

        # The allocator depends on the environment at the unpickling site, so
        # we don't pickle it
        state['_allocator'] = None

        return state

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

        self._allocator = default_allocator(
            f'{type(self._compiler).__name__}.{self._language}.{self._platform}'
        )


# *** Recursive compilation ("rcompile") machinery


class RCompiles(CacheInstances):

    """
    A cache for abstract Callables obtained from lowering expressions.
    Here, "abstract Callable" means that any user-level symbolic object appearing
    in the input expressions is replaced by a corresponding abstract object.
    """

    _instance_cache_size = None

    def __init__(self, exprs, cls):
        self.exprs = exprs
        self.cls = cls

        # NOTE: Constructed lazily at `__call__` time because `**kwargs` is
        # unhashable for historical reasons (e.g., Compiler objects are mutable,
        # though in practice they are unique per Operator, so only "locally"
        # mutable)
        self._output = None

    def compile(self, **kwargs):
        if self._output is None:
            self._output = self.cls._lower(self.exprs, **kwargs)
        return self._output


# Default action (perform or bypass) for selected compilation passes upon
# recursive compilation
# NOTE: it may not only be pointless to apply the following passes recursively
# (because once, during the main compilation phase, is simply enough), but also
# dangerous as some of them (the minority) might break in some circumstances
# if applied in cascade (e.g., `linearization` on top of `linearization`)
rcompile_registry = {
    'avoid_denormals': False,
    'linearize': False,
    'place-transfers': False
}


def rcompile(expressions, kwargs, options, target=None):
    """
    Perform recursive compilation on an ordered sequence of symbolic expressions.
    """
    expressions = as_tuple(expressions)
    options = {**options, **rcompile_registry}

    if target is None:
        cls = operator_selector(**kwargs)
        kwargs['options'] = options
    else:
        kwargs = parse_kwargs(**target)
        cls = operator_selector(**kwargs)
        kwargs = cls._normalize_kwargs(**kwargs)
        kwargs['options'].update(options)

    # Recursive profiling not supported -- would be a complete mess
    kwargs.pop('profiler', None)

    # Recursive compilation is expensive, so we cache the result because sometimes
    # it is called multiple times for the same input
    irs, byproduct0 = RCompiles(expressions, cls).compile(**kwargs)

    key = lambda i: isinstance(i, (EntryFunction, DeviceFunction))
    byproduct = byproduct0.filter(key)

    return irs, byproduct


# *** Misc helpers


IRs = namedtuple('IRs', 'expressions clusters stree uiet iet')


class ArgumentsMap(dict):

    def __init__(self, args, grid, op):
        super().__init__(args)

        self.grid = grid
        self.op = op

    @property
    def comm(self):
        """The MPI communicator the arguments are collective over."""
        return self.grid.comm if self.grid is not None else MPI.COMM_NULL

    @property
    def opkwargs(self):
        temp_registry = {v: k for k, v in compiler_registry.items()}
        compiler = temp_registry[self.compiler.__class__]

        return {'platform': self.platform.name,
                'compiler': compiler,
                'language': self.language}

    @property
    def allocator(self):
        return self.op._allocator

    @property
    def platform(self):
        return self.op._platform

    @property
    def language(self):
        return self.op._language

    @property
    def compiler(self):
        return self.op._compiler

    @property
    def options(self):
        return self.op._options

    @property
    def saved_mapper(self):
        """
        The number of saved TimeFunctions in the Operator, grouped by
        memory hierarchy layer.
        """
        key0 = lambda f: (f.is_TimeFunction and
                          f.save is not None and
                          not isinstance(f.save, Buffer))
        functions = [f for f in self.op.input if key0(f)]

        key1 = lambda f: f.layer
        mapper = as_mapper(functions, key1)

        return mapper

    @cached_property
    def _op_symbols(self):
        """Symbols in the Operator which may or may not carry data"""
        return FindSymbols().visit(self.op)

    @cached_property
    def _op_functions(self):
        """Function symbols in the Operator"""
        return [i for i in self._op_symbols if i.is_DiscreteFunction and not i.alias]

    def _apply_override(self, i):
        try:
            return self.get(i.name, i)._obj
        except AttributeError:
            return self.get(i.name, i)

    def _get_nbytes(self, i):
        """
        Extract the allocated size of a symbol, accounting for any
        overrides.
        """
        obj = self._apply_override(i)
        try:
            # Non-regular AbstractFunction (compressed, etc)
            nbytes = obj.nbytes_max
        except (AttributeError, ValueError):
            # Either garden-variety AbstractFunction, or uninitialised
            # function used in estimate. In the latter case, fall back
            # to nbytes, as it is typically zero
            nbytes = obj.nbytes

        # Could nominally have symbolic nbytes at this point
        if isinstance(nbytes, sympy.Basic):
            return subs_op_args(nbytes, self)

        return nbytes

    @cached_property
    def nbytes_avail_mapper(self):
        """
        The amount of memory available after accounting for the memory
        consumed by the Operator, in bytes, grouped by memory hierarchy layer.
        """
        mapper = {}

        # The amount of space available on the disk
        usage = shutil.disk_usage(gettempdir())
        mapper[disk_layer] = usage.free

        # The amount of space available on the device
        if isinstance(self.platform, Device):
            deviceid = max(self.get('deviceid', 0), 0)
            mapper[device_layer] = self.platform.memavail(deviceid=deviceid)

        # The amount of space available on the host
        try:
            nproc = self.grid.distributor.nprocs_local
        except AttributeError:
            nproc = 1
        mapper[host_layer] = int(ANYCPU.memavail() / nproc)

        for layer in (host_layer, device_layer):
            try:
                mapper[layer] -= self.nbytes_consumed_operator.get(layer, 0)
            except KeyError:  # Might not have this layer in the mapper
                pass

        mapper = {k: int(v) for k, v in mapper.items()}

        return mapper

    @cached_property
    def nbytes_consumed(self):
        """Memory consumed by all objects in the Operator"""
        mem_locations = (
            self.nbytes_consumed_functions,
            self.nbytes_consumed_arrays,
            self.nbytes_consumed_memmapped
        )
        return {layer: sum(loc[layer] for loc in mem_locations) for layer in _layers}

    @cached_property
    def nbytes_consumed_operator(self):
        """Memory consumed by objects allocated within the Operator"""
        mem_locations = (
            self.nbytes_consumed_arrays,
            self.nbytes_consumed_memmapped
        )
        return {layer: sum(loc[layer] for loc in mem_locations) for layer in _layers}

    @cached_property
    def nbytes_consumed_functions(self):
        """
        Memory consumed on both device and host by Functions in the
        corresponding Operator.
        """
        host = 0
        device = 0
        # Filter out arrays, aliases and non-AbstractFunction objects
        for i in self._op_functions:
            v = self._get_nbytes(i)
            if i._mem_host or i._mem_mapped:
                # No need to add to device , as it will be counted
                # by nbytes_consumed_memmapped
                host += v
            elif i._mem_local:
                if isinstance(self.platform, Device):
                    device += v
                else:
                    host += v

        return {disk_layer: 0, host_layer: host, device_layer: device}

    @cached_property
    def nbytes_consumed_arrays(self):
        """
        Memory consumed on both device and host by C-land Arrays
        in the corresponding Operator.
        """
        host = 0
        device = 0
        # Temporaries such as Arrays are allocated and deallocated on-the-fly
        # while in C land, so they need to be accounted for as well
        for i in self._op_symbols:
            if not i.is_Array or not i._mem_heap or i.alias \
               or not i.is_regular:
                continue

            if i.is_regular:
                nbytes = i.nbytes
            else:
                nbytes = i.nbytes_max
            v = subs_op_args(nbytes, self)
            if not is_integer(v):
                # E.g. the Arrays used to store the MPI halo exchanges
                continue

            if i._mem_host:
                host += v
            elif i._mem_local:
                if isinstance(self.platform, Device):
                    device += v
                else:
                    host += v
            elif i._mem_mapped:
                if isinstance(self.platform, Device):
                    device += v
                host += v

        return {disk_layer: 0, host_layer: host, device_layer: device}

    @cached_property
    def nbytes_consumed_memmapped(self):
        """
        Memory also consumed on device by data which is to be memcpy-d
        from host to device at the start of computation.
        """
        device = 0
        # All input Functions are yet to be memcpy-ed to the device
        # TODO: this may not be true depending on `devicerm`, which is however
        # virtually never used
        if isinstance(self.platform, Device):
            for i in self.op.input:
                if not is_on_device(i, self.options['gpu-fit']):
                    continue
                try:
                    if i._mem_mapped:
                        device += self._get_nbytes(i)
                except AttributeError:
                    pass

        return {disk_layer: 0, host_layer: 0, device_layer: device}

    @cached_property
    def nbytes_snapshots(self):
        disk = 0
        # Layers are sometimes aliases, so include aliases here
        for i in self._op_symbols:
            try:
                if i._child is None and i.alias is not True:
                    # Use only the "innermost" layer to avoid counting snapshots
                    # twice. This layer will have no child.
                    v = self._apply_override(i)
                    disk += v.size_snapshot*v._time_size_ideal*np.dtype(v.dtype).itemsize
            except AttributeError:
                pass

        return {disk_layer: disk, host_layer: 0, device_layer: 0}


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
            warning(f"Setting `opt={str(dle)}`")
            opt = dle
    elif 'opt' in kwargs:
        opt = kwargs.pop('opt')
    else:
        opt = configuration['opt']

    if not opt or isinstance(opt, str):
        mode, options = opt, {}
        # Legacy Operator(..., opt='openmp', ...) support
        if mode == 'openmp':
            mode = 'noop'
            options = {'openmp': True}
    elif isinstance(opt, tuple):
        if len(opt) == 0:
            mode, options = 'noop', {}
        elif isinstance(opt[-1], (dict, frozendict)):
            if len(opt) == 2:
                mode, options = opt
            else:
                mode, options = tuple(flatten(i.split(',') for i in opt[:-1])), opt[-1]
        else:
            mode, options = tuple(flatten(i.split(',') for i in opt)), {}
    else:
        raise InvalidOperator(f"Illegal `opt={str(opt)}`")

    # `openmp` in mode e.g `opt=('openmp', 'simd', {})`
    if mode and 'openmp' in mode:
        options['openmp'] = True
        mode = tuple(i for i in as_tuple(mode) if i != 'openmp')

    # `opt`, deprecated kwargs
    kwopenmp = kwargs.get('openmp', options.get('openmp'))
    if kwopenmp is None:
        openmp = 'openmp' in kwargs.get('language', configuration['language'])
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
            warning(f"Ignoring deprecated optimization option `{i}`")
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
            raise InvalidOperator(f"Illegal `platform={str(platform)}`")
        kwargs['platform'] = platform_registry[platform]()
    else:
        kwargs['platform'] = configuration['platform']

    # `language`
    language = kwargs.get('language')
    if language is not None:
        if not isinstance(language, str):
            raise ValueError("Argument `language` should be a `str`")
        if language not in configuration._accepted['language']:
            raise InvalidOperator(f"Illegal `language={str(language)}`")
        kwargs['language'] = language
    elif kwopenmp is not None:
        # Handle deprecated `openmp` kwarg for backward compatibility
        omp = {'C': 'openmp', 'CXX': 'CXXopenmp'}.get(configuration['language'],
                                                      'openmp')
        kwargs['language'] = omp if openmp else 'C'
    else:
        kwargs['language'] = configuration['language']

    # `compiler`
    compiler = kwargs.get('compiler')
    if compiler is not None:
        if not isinstance(compiler, str):
            raise ValueError("Argument `compiler` should be a `str`")
        if compiler not in configuration._accepted['compiler']:
            raise InvalidOperator(f"Illegal `compiler={str(compiler)}`")
        kwargs['compiler'] = compiler_registry[compiler](platform=kwargs['platform'],
                                                         language=kwargs['language'],
                                                         mpi=configuration['mpi'],
                                                         name=compiler)
    elif any([platform, language]):
        kwargs['compiler'] =\
            configuration['compiler'].__new_with__(platform=kwargs['platform'],
                                                   language=kwargs['language'],
                                                   mpi=configuration['mpi'])
    else:
        kwargs['compiler'] = configuration['compiler'].__new_with__()

    # Make sure compiler and language are compatible
    if compiler is not None and kwargs['compiler']._cpp and \
            kwargs['language'] in ['C', 'openmp']:
        kwargs['language'] = 'CXX' if kwargs['language'] == 'C' else 'CXXopenmp'
    if 'CXX' in kwargs['language'] and not kwargs['compiler']._cpp:
        kwargs['compiler'] = kwargs['compiler'].__new_with__(cpp=True)

    # `allocator`
    kwargs['allocator'] = default_allocator(
        f"{kwargs['compiler'].__class__.__name__}"
        f".{kwargs['language']}"
        f".{kwargs['platform']}"
    )

    # Normalize `subs`, if any
    kwargs['subs'] = {k: sympify(v) for k, v in kwargs.get('subs', {}).items()}

    return kwargs

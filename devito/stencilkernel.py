from __future__ import absolute_import

import operator
from collections import OrderedDict, namedtuple
from ctypes import c_double, c_int
from functools import reduce
from hashlib import sha1
from itertools import combinations
from os import path

import cgen as c
import numpy as np

from devito.cgen_utils import Allocator
from devito.compiler import (get_compiler_from_env, get_tmp_dir,
                             jit_compile_and_load)
from devito.dimension import BufferedDimension, Dimension, time
from devito.dle import compose_nodes, filter_iterations, transform
from devito.dse import (estimate_cost, estimate_memory, indexify, rewrite)
from devito.interfaces import SymbolicData, Forward, Backward
from devito.logger import bar, error, info, info_at
from devito.nodes import (Element, Expression, Function, Iteration, List,
                          LocalExpression, TimedList)
from devito.profiler import Profiler
from devito.stencil import Stencil
from devito.tools import as_tuple, filter_ordered, filter_sorted, flatten, partial_order
from devito.visitors import (FindNodes, FindSections, FindSymbols, FindScopes,
                             IsPerfectIteration, MergeOuterIterations,
                             ResolveIterationVariable, SubstituteExpression, Transformer)

__all__ = ['StencilKernel']


class StencilKernel(Function):

    """
    Cache of auto-tuned StencilKernels.
    """
    _AT_cache = {}

    _default_headers = ['#define _POSIX_C_SOURCE 200809L']
    _default_includes = ['stdlib.h', 'math.h', 'sys/time.h']

    """A special :class:`Function` to evaluate expressions through just-in-time
    compilation of C code.

    :param expressions: SymPy equation or list of equations that define the
                        the kernel of this Operator.
    :param kwargs: Accept the following entries: ::

        * name : Name of the kernel function - defaults to "Kernel".
        * subs : Dict or list of dicts containing SymPy symbol substitutions
                 for each expression respectively.
        * time_axis : :class:`TimeAxis` object to indicate direction in which
                      to advance time during computation.
        * dse : Use the Devito Symbolic Engine to optimize the expressions -
                defaults to "advanced".
        * dle : Use the Devito Loop Engine to optimize the loops -
                defaults to "advanced".
        * compiler: Compiler class used to perform JIT compilation.
                    If not provided, the compiler will be inferred from the
                    environment variable DEVITO_ARCH, or default to GNUCompiler.
        * profiler: :class:`devito.Profiler` instance to collect profiling
                    meta-data at runtime. Use profiler=None to disable profiling.
    """
    def __init__(self, expressions, **kwargs):
        name = kwargs.get("name", "Kernel")
        subs = kwargs.get("subs", {})
        time_axis = kwargs.get("time_axis", Forward)
        dse = kwargs.get("dse", "advanced")
        dle = kwargs.get("dle", "advanced")
        compiler = kwargs.get("compiler", None)

        # Default attributes required for compilation
        self.compiler = compiler or get_compiler_from_env()
        self.profiler = kwargs.get("profiler", Profiler(self.compiler.openmp))
        self._headers = list(self._default_headers)
        self._includes = list(self._default_includes)
        self._lib = None
        self._cfunction = None

        # Set the direction of time acoording to the given TimeAxis
        time.reverse = time_axis == Backward

        # Normalize the expressions
        expressions = [indexify(s) for s in as_tuple(expressions)]
        expressions = [s.xreplace(subs) for s in expressions]

        # Retrieve the data type of the StencilKernel
        self.dtype = self._retrieve_dtype(expressions)

        # Apply the Devito Symbolic Engine for symbolic optimization
        dse_state = rewrite(expressions, mode=dse)

        # Wrap expressions with Iterations according to dimensions
        nodes = self._schedule_expressions(dse_state)

        # Introduce C-level profiling infrastructure
        self.sections = OrderedDict()
        nodes = self._profile_sections(nodes)

        # Parameters of the StencilKernel (Dimensions necessary for data casts)
        parameters = FindSymbols('kernel-data').visit(nodes)
        dimensions = FindSymbols('dimensions').visit(nodes)
        dimensions += [d.parent for d in dimensions if d.is_Buffered]
        parameters += filter_ordered([d for d in dimensions if d.size is None],
                                     key=operator.attrgetter('name'))

        # Resolve and substitute dimensions for loop index variables
        subs = {}
        nodes = ResolveIterationVariable().visit(nodes, subs=subs)
        nodes = SubstituteExpression(subs=subs).visit(nodes)

        # Apply the Devito Loop Engine for loop optimization
        dle_state = transform(nodes, set_dle_mode(dle, self.compiler), self.compiler)
        parameters += [i.argument for i in dle_state.arguments]
        self._includes.extend(list(dle_state.includes))

        # Introduce all required C declarations
        nodes, elemental_functions = self._insert_declarations(dle_state, parameters)
        self.elemental_functions = elemental_functions

        # Track the DSE and DLE output, as they may be useful later
        self._dse_state = dse_state
        self._dle_state = dle_state

        # Finish instantiation
        super(StencilKernel, self).__init__(name, nodes, 'int', parameters, ())

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply defined stencil kernel to a set of data objects"""
        if len(args) <= 0:
            args = self.parameters

        # Perform auto-tuning if the user requests it and loop blocking is in use
        maybe_autotune = kwargs.get('autotune', False)

        # Map of required arguments and actual dimension sizes
        arguments = OrderedDict([(arg.name, arg) for arg in self.parameters])
        dim_sizes = {}

        # Traverse positional args and infer loop sizes for open dimensions
        f_args = [f for f in arguments.values() if isinstance(f, SymbolicData)]
        for f, arg in zip(f_args, args):
            # Ensure we're dealing or deriving numpy arrays
            data = f.data if isinstance(f, SymbolicData) else arg
            if not isinstance(data, np.ndarray):
                error('No array data found for argument %s' % f.name)
            arguments[f.name] = data

            # Ensure data dimensions match symbol dimensions
            for i, dim in enumerate(f.indices):
                # Infer open loop limits
                if dim.size is None:
                    # First, try to find dim size in kwargs
                    if dim.name in kwargs:
                        dim_sizes[dim] = kwargs[dim.name]

                    if dim in dim_sizes:
                        # Ensure size matches previously defined size
                        if not dim.is_Buffered:
                            assert dim_sizes[dim] <= data.shape[i]
                    else:
                        # Derive size from grid data shape and store
                        dim_sizes[dim] = data.shape[i]

                    # Ensure parent for buffered dims is defined
                    if dim.is_Buffered and dim.parent not in dim_sizes:
                        dim_sizes[dim.parent] = dim_sizes[dim]
                else:
                    if not isinstance(dim, BufferedDimension):
                        assert dim.size == data.shape[i]

        # Add user-provided block sizes, if any
        dle_arguments = OrderedDict()
        for i in self._dle_state.arguments:
            dim_size = dim_sizes.get(i.original_dim, i.original_dim.size)
            assert dim_size is not None, "Unable to match arguments and values"
            if i.value:
                try:
                    dle_arguments[i.argument] = i.value(dim_size)
                except TypeError:
                    dle_arguments[i.argument] = i.value
                    # User-provided block size available, do not autotune
                    maybe_autotune = False
            else:
                dle_arguments[i.argument] = dim_size
        dim_sizes.update(dle_arguments)

        # Insert loop size arguments from dimension values
        d_args = [d for d in arguments.values() if isinstance(d, Dimension)]
        for d in d_args:
            arguments[d.name] = dim_sizes[d]

        # Might have been asked to auto-tune the block size
        if maybe_autotune:
            self._autotune(arguments)

        # Add profiler structs
        if self.profiler:
            cpointer = self.profiler.as_ctypes_pointer(Profiler.TIME)
            arguments[self.profiler.s_name] = cpointer

        # Invoke kernel function with args
        self.cfunction(*list(arguments.values()))

        # Output summary of performance achieved
        summary = self._profile_summary(dim_sizes)
        with bar():
            for k, v in summary.items():
                name = '%s<%s>' % (k, ','.join('%d' % i for i in v.itershape))
                info("Section %s with OI=%.2f computed in %.3f s [Perf: %.2f GFlops/s]" %
                     (name, v.oi, v.time, v.gflopss))

        return summary

    def _profile_sections(self, nodes):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        mapper = {}
        for i, expr in enumerate(nodes):
            for itspace in FindSections().visit(expr).keys():
                for j in itspace:
                    if IsPerfectIteration().visit(j) and j not in mapper:
                        # Insert `TimedList` block. This should come from
                        # the profiler, but we do this manually for now.
                        lname = 'loop_%s_%d' % (j.index, i)
                        mapper[j] = TimedList(gname=self.profiler.t_name,
                                              lname=lname, body=j)
                        self.profiler.t_fields += [(lname, c_double)]

                        # Estimate computational properties of the timed section
                        # (operational intensity, memory accesses)
                        expressions = FindNodes(Expression).visit(j)
                        ops = estimate_cost([e.expr for e in expressions])
                        memory = estimate_memory([e.expr for e in expressions])
                        self.sections[itspace] = Profile(lname, ops, memory)
                        break
        processed = Transformer(mapper).visit(List(body=nodes))
        return processed

    def _profile_summary(self, dim_sizes):
        """
        Produce a summary of the performance achieved
        """
        summary = PerformanceSummary()
        for itspace, profile in self.sections.items():
            dims = {i: i.dim.parent if i.dim.is_Buffered else i.dim for i in itspace}

            # Time
            time = self.profiler.timings[profile.timer]

            # Flops
            itershape = [i.extent(finish=dim_sizes.get(dims[i])) for i in itspace]
            iterspace = reduce(operator.mul, itershape)
            flops = float(profile.ops*iterspace)
            gflops = flops/10**9

            # Compulsory traffic
            datashape = [i.dim.size or dim_sizes[dims[i]] for i in itspace]
            dataspace = reduce(operator.mul, datashape)
            traffic = profile.memory*dataspace*self.dtype().itemsize

            # Derived metrics
            oi = flops/traffic
            gflopss = gflops/time

            # Keep track of performance achieved
            summary.setsection(profile.timer, time, gflopss, oi, itershape, datashape)

        # Rename the most time consuming section as 'main'
        summary['main'] = summary.pop(max(summary, key=summary.get))

        return summary

    def _autotune(self, arguments):
        """Use auto-tuning on this StencilKernel to determine empirically the
        best block sizes (when loop blocking is in use). The block sizes tested
        are those listed in ``options['at_blocksizes']``, plus the case that is
        as if blocking were not applied (ie, unitary block size)."""
        if not self._dle_state.has_applied_blocking:
            return

        at_arguments = arguments.copy()

        # Output data must not be changed
        output = [i.base.label.name for i in self._dse_state.output_fields]
        for k, v in arguments.items():
            if k in output:
                at_arguments[k] = v.copy()

        # Squeeze dimensions to minimize auto-tuning time
        iterations = FindNodes(Iteration).visit(self.body)
        squeezable = [i.dim.parent.name for i in iterations
                      if i.is_Sequential and i.dim.is_Buffered]

        # Attempted block sizes
        mapper = OrderedDict([(i.argument.name, i) for i in self._dle_state.arguments])
        blocksizes = [OrderedDict([(i, v) for i in mapper])
                      for v in options['at_blocksize']]
        if self._dle_state.needs_aggressive_autotuning:
            elaborated = []
            for blocksize in list(blocksizes)[:3]:
                for i in list(blocksizes):
                    handle = i.items()[-1]
                    elaborated.append(OrderedDict(blocksize.items()[:-1] + [handle]))
            for blocksize in list(blocksizes):
                ncombs = len(blocksize)
                for i in range(ncombs):
                    for j in combinations(blocksize, i+1):
                        handle = [(k, blocksize[k]*2 if k in j else v)
                                  for k, v in blocksize.items()]
                        elaborated.append(OrderedDict(handle))
            blocksizes.extend(elaborated)

        # Note: there is only a single loop over 'blocksize' because only
        # square blocks are tested
        timings = OrderedDict()
        for blocksize in blocksizes:
            illegal = False
            for k, v in at_arguments.items():
                if k in blocksize:
                    val = blocksize[k]
                    handle = at_arguments.get(mapper[k].original_dim.name)
                    if val <= mapper[k].iteration.end(handle):
                        at_arguments[k] = val
                    else:
                        # Block size cannot be larger than actual dimension
                        illegal = True
                        break
                elif k in squeezable:
                    at_arguments[k] = options['at_squeezer']
            if illegal:
                continue

            # Add profiler structs
            if self.profiler:
                cpointer = self.profiler.as_ctypes_pointer(Profiler.TIME)
                at_arguments[self.profiler.s_name] = cpointer

            self.cfunction(*list(at_arguments.values()))
            elapsed = sum(self.profiler.timings.values())
            timings[tuple(blocksize.items())] = elapsed
            info_at("<%s>: %f" %
                    (','.join('%d' % i for i in blocksize.values()), elapsed))

        best = dict(min(timings, key=timings.get))
        for k, v in arguments.items():
            if k in mapper:
                arguments[k] = best[k]

        info('Auto-tuned block shape: %s' % best)

    def _schedule_expressions(self, dse_state):
        """Wrap :class:`Expression` objects within suitable hierarchies of
        :class:`Iteration` according to dimensions and stencils."""

        # Establish a partial ordering for the Iterations based on the order
        # by which dimensions appeared in the input expressions
        ordering = tuple(flatten(list(Stencil(i)) for i in dse_state.input))
        ordering = list(OrderedDict(zip(ordering, ordering)))

        processed = []
        schedule = OrderedDict()
        for c in dse_state.clusters:
            # Build the Expression objects to be inserted within an Iteration tree
            expressions = [Expression(v, np.int32 if c.trace.is_index(k) else self.dtype)
                           for k, v in c.trace.items()]

            # Filter out aliasing due to buffered dimensions
            key = lambda d: d.parent if d.is_Buffered else d
            dimensions = filter_ordered(list(c.stencil.keys()), key=key)

            # Reorder the expressions based on the global partial ordering
            dimensions = filter_sorted(dimensions, key=lambda d: ordering.index(d))
            stencil = Stencil([(key(d), c.stencil.get(key(d))) for d in dimensions])

            if not stencil.empty:
                root = None
                entries = stencil.entries

                # Can I reuse any of the previously scheduled Iterations ?
                for index, i in enumerate(entries):
                    if i not in schedule:
                        break
                    root = schedule[i]
                needed = entries[index:]

                # Build and insert the required Iterations
                iters = [Iteration([], i.dim, i.dim.size, offsets=i.ofs) for i in needed]
                body, tree = compose_nodes(iters + [expressions], retrieve=True)
                scheduling = OrderedDict(zip(needed, tree))
                if root is None:
                    processed.append(body)
                    schedule = scheduling
                else:
                    nodes = list(root.nodes) + [body]
                    mapper = {root: root._rebuild(nodes, **root.args_frozen)}
                    transformer = Transformer(mapper)
                    processed = list(transformer.visit(processed))
                    schedule = OrderedDict(schedule.items()[:index] + scheduling.items())
                    for k, v in list(schedule.items()):
                        schedule[k] = transformer.rebuilt.get(v, v)
            else:
                # No Iterations are needed
                processed.extend(expressions)

        return processed

    def _insert_declarations(self, dle_state, parameters):
        """Populate the StencilKernel's body with the required array and
        variable declarations, to generate a legal C file."""

        nodes = dle_state.nodes

        # Resolve function calls first
        scopes = []
        for k, v in FindScopes().visit(nodes).items():
            if k.is_FunCall:
                function = dle_state.func_table[k.name]
                scopes.extend(FindScopes().visit(function, queue=list(v)).items())
            else:
                scopes.append((k, v))

        # Determine all required declarations
        allocator = Allocator()
        mapper = OrderedDict()
        for k, v in scopes:
            if k.is_scalar:
                # Inline declaration
                mapper[k] = LocalExpression(**k.args)
            elif k.output_function._mem_external:
                # Nothing to do, variable passed as kernel argument
                continue
            elif k.output_function._mem_stack:
                # On the stack, as established by the DLE
                key = lambda i: i.dim not in k.output_function.indices
                site = filter_iterations(v, key=key, stop='consecutive')
                allocator.push_stack(site[-1], k.output_function)
            else:
                # On the heap, as a tensor that must be globally accessible
                allocator.push_heap(k.output_function)

        # Introduce declarations on the stack
        for k, v in allocator.onstack:
            allocs = as_tuple([Element(i) for i in v])
            mapper[k] = Iteration(allocs + k.nodes, **k.args_frozen)
        nodes = Transformer(mapper).visit(nodes)
        elemental_functions = Transformer(mapper).visit(dle_state.elemental_functions)

        # Introduce declarations on the heap (if any)
        if allocator.onheap:
            decls, allocs, frees = zip(*allocator.onheap)
            nodes = List(header=decls + allocs, body=nodes, footer=frees)

        return nodes, elemental_functions

    def _retrieve_dtype(self, expressions):
        """
        Retrieve the data type of a set of expressions. Raise an error if there
        is no common data type (ie, if at least one expression differs in the
        data type).
        """
        lhss = set([s.lhs.base.function.dtype for s in expressions])
        if len(lhss) != 1:
            raise RuntimeError("Expression types mismatch.")
        return lhss.pop()

    @property
    def _cparameters(self):
        cparameters = super(StencilKernel, self)._cparameters
        cparameters += [c.Pointer(c.Value('struct %s' % self.profiler.s_name,
                                          self.profiler.t_name))]
        return cparameters

    @property
    def ccode(self):
        """Returns the C code generated by this kernel.

        This function generates the internal code block from Iteration
        and Expression objects, and adds the necessary template code
        around it.
        """
        blankline = c.Line("")

        # Generate function body with all the trimmings
        body = [e.ccode for e in self.body]
        ret = [c.Statement("return 0")]
        kernel = c.FunctionBody(self._ctop, c.Block(self._ccasts + body + ret))

        # Generate elemental functions produced by the DLE
        elemental_functions = [e.ccode for e in self.elemental_functions]
        elemental_functions += [blankline]

        # Generate file header with includes and definitions
        header = [c.Line(i) for i in self._headers]
        includes = [c.Include(i, system=False) for i in self._includes]
        includes += [blankline]
        profiling = [self.profiler.as_cgen_struct(Profiler.TIME), blankline]
        return c.Module(header + includes + profiling + elemental_functions + [kernel])

    @property
    def cfunction(self):
        """Returns the JIT-compiled C function as a ctypes.FuncPtr object

        Note that this invokes the JIT compilation toolchain with the
        compiler class derived in the constructor

        :returns: The generated C function
        """
        if self._lib is None:
            ccode = self.ccode
            hash_key = sha1(str(ccode).encode()).hexdigest()
            basename = path.join(get_tmp_dir(), hash_key)
            self._lib = jit_compile_and_load(ccode, basename, self.compiler)
        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            self._cfunction.argtypes = self.argtypes

        return self._cfunction

    @property
    def argtypes(self):
        """Create argument types for defining function signatures via ctypes

        :returns: A list of ctypes of the matrix parameters and scalar parameters
        """
        return [c_int if isinstance(v, Dimension) else
                np.ctypeslib.ndpointer(dtype=v.dtype, flags='C')
                for v in self.parameters]


# Helpers for performance tracking

"""
A helper to return structured performance data.
"""
PerfEntry = namedtuple('PerfEntry', 'time gflopss oi itershape datashape')


class PerformanceSummary(OrderedDict):

    """
    A special dictionary to track and view performance data.
    """

    def setsection(self, key, time, gflopss, oi, itershape, datashape):
        self[key] = PerfEntry(time, gflopss, oi, itershape, datashape)

    @property
    def gflopss(self):
        return OrderedDict([(k, v.gflopss) for k, v in self.items()])

    @property
    def oi(self):
        return OrderedDict([(k, v.oi) for k, v in self.items()])

    @property
    def timings(self):
        return OrderedDict([(k, v.time) for k, v in self.items()])


# StencilKernel options and name conventions

"""
A dict of standard names to be used for code generation
"""
cnames = {
    'loc_timer': 'loc_timer',
    'glb_timer': 'glb_timer'
}

"""
StencilKernel options
"""
options = {
    'at_squeezer': 3,
    'at_blocksize': [8, 16, 24, 32, 40, 64, 128]
}

"""
A helper to track profiled sections of code.
"""
Profile = namedtuple('Profile', 'timer ops memory')


# Helpers to use a StencilKernel

def set_dle_mode(mode, compiler):
    """
    Transform :class:`StencilKernel` input in a format understandable by the DLE.
    """
    if not mode:
        return 'noop'
    mode = as_tuple(mode)
    params = mode[-1]
    if isinstance(params, dict):
        params['openmp'] = compiler.openmp
    else:
        params = {'openmp': compiler.openmp}
        mode += (params,)
    return mode

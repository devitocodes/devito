from __future__ import absolute_import

import operator
from collections import OrderedDict, namedtuple
from ctypes import c_int
from functools import reduce

import cgen as c
import numpy as np
import sympy

from devito.autotuning import autotune
from devito.cgen_utils import Allocator, blankline
from devito.compiler import jit_compile, load
from devito.dimension import Dimension, time
from devito.dle import compose_nodes, filter_iterations, transform
from devito.dse import (clusterize, estimate_cost, estimate_memory, indexify,
                        rewrite, q_indexed)
from devito.interfaces import SymbolicData, Forward, Backward
from devito.logger import bar, error, info
from devito.nodes import (Element, Expression, Function, Iteration, List,
                          LocalExpression, TimedList)
from devito.parameters import configuration
from devito.profiling import Profiler
from devito.stencil import Stencil
from devito.tools import as_tuple, filter_ordered, flatten
from devito.visitors import (FindNodes, FindSections, FindSymbols, FindScopes,
                             IsPerfectIteration, ResolveIterationVariable,
                             SubstituteExpression, Transformer)
from devito.exceptions import InvalidArgument, InvalidOperator

__all__ = ['Operator']


class OperatorBasic(Function):

    _default_headers = ['#define _POSIX_C_SOURCE 200809L']
    _default_includes = ['stdlib.h', 'math.h', 'sys/time.h']

    """A special :class:`Function` to generate and compile C code evaluating
    an ordered sequence of stencil expressions.

    :param expressions: SymPy equation or list of equations that define the
                        the kernel of this Operator.
    :param kwargs: Accept the following entries: ::

        * name : Name of the kernel function - defaults to "Kernel".
        * subs : Dict or list of dicts containing SymPy symbol substitutions
                 for each expression respectively.
        * time_axis : :class:`TimeAxis` object to indicate direction in which
                      to advance time during computation.
        * dse : Use the Devito Symbolic Engine to optimize the expressions -
                defaults to ``configuration['dse']``.
        * dle : Use the Devito Loop Engine to optimize the loops -
                defaults to ``configuration['dle']``.
    """
    def __init__(self, expressions, **kwargs):
        expressions = as_tuple(expressions)

        # Input check
        if any(not isinstance(i, sympy.Eq) for i in expressions):
            raise InvalidOperator("Only SymPy expressions are allowed.")

        self.name = kwargs.get("name", "Kernel")
        subs = kwargs.get("subs", {})
        time_axis = kwargs.get("time_axis", Forward)
        dse = kwargs.get("dse", configuration['dse'])
        dle = kwargs.get("dle", configuration['dle'])

        # Default attributes required for compilation
        self._headers = list(self._default_headers)
        self._includes = list(self._default_includes)
        self._lib = None
        self._cfunction = None

        # Set the direction of time acoording to the given TimeAxis
        time.reverse = time_axis == Backward

        # Expression lowering
        expressions = [indexify(s) for s in expressions]
        expressions = [s.xreplace(subs) for s in expressions]

        # Analysis 1 - required *also after* the Operator construction
        self.dtype = self._retrieve_dtype(expressions)
        self.output = self._retrieve_output_fields(expressions)

        # Analysis 2 - required *for* the Operator construction
        ordering = self._retrieve_loop_ordering(expressions)
        stencils = self._retrieve_stencils(expressions)

        # Group expressions based on their Stencil
        clusters = clusterize(expressions, stencils)

        # Apply the Devito Symbolic Engine for symbolic optimization
        clusters = rewrite(clusters, mode=dse)

        # Wrap expressions with Iterations according to dimensions
        nodes = self._schedule_expressions(clusters, ordering)

        # Introduce C-level profiling infrastructure
        self.sections = OrderedDict()
        nodes = self._profile_sections(nodes)

        # Parameters of the Operator (Dimensions necessary for data casts)
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
        dle_state = transform(nodes, *set_dle_mode(dle))
        parameters += [i.argument for i in dle_state.arguments]
        self._includes.extend(list(dle_state.includes))

        # Introduce all required C declarations
        nodes, elemental_functions = self._insert_declarations(dle_state, parameters)
        self.elemental_functions = elemental_functions

        # Track the DLE output, as it might be useful at execution time
        self._dle_state = dle_state

        # Finish instantiation
        super(OperatorBasic, self).__init__(self.name, nodes, 'int', parameters, ())

    def arguments(self, *args, **kwargs):
        """
        Return the arguments necessary to apply the Operator.
        """
        if len(args) == 0:
            args = self.parameters

        # Will perform auto-tuning if the user requested it and loop blocking was used
        maybe_autotune = kwargs.get('autotune', False)

        # Initialise argument map and a map of dimension names to values
        arguments = OrderedDict([(arg.name, arg) for arg in self.parameters])
        dim_sizes = dict([(arg.name, arg.size) for arg in self.parameters
                          if isinstance(arg, Dimension)])

        # Override explicitly provided dim sizes from **kwargs
        for name, value in kwargs.items():
            if name in dim_sizes:
                dim_sizes[name] = value

        # Have we been provided substitutes for symbol data?
        # Only SymbolicData can be overridden with this route
        r_args = [f_n for f_n, f in arguments.items() if isinstance(f, SymbolicData)]
        o_vals = OrderedDict([arg for arg in kwargs.items() if arg[0] in r_args])

        # Are any passed symbols composite? i.e. are they composed of other symbols?
        for sym in o_vals.values():
            if isinstance(sym, SymbolicData) and sym.is_CompositeData:
                o_vals.update(OrderedDict([(c.name, c) for c in sym.children]))

        # Replace the overridden values with the provided ones
        for argname in o_vals.keys():
            arguments[argname] = o_vals[argname]

        # Traverse positional args and infer loop sizes for open dimensions
        f_args = [(name, f) for name, f in arguments.items()
                  if isinstance(f, SymbolicData)]
        for fname, f in f_args:
            arguments[fname] = self._arg_data(f)
            shape = self._arg_shape(f)

            # Ensure data dimensions match symbol dimensions
            for i, dim in enumerate(f.indices):
                # We don't need to check sizes for buffered dimensions
                # against data shapes, all we need is the size of the parent.
                if dim.is_Buffered:
                    continue

                # Check data sizes for dimensions with a fixed size
                if dim.size is not None:
                    if not shape[i] <= dim.size:
                        error('Size of data argument for %s is greater than the size '
                              'of dimension %s: %d' % (fname, dim.name, dim.size))
                        raise InvalidArgument('Wrong data shape encountered')
                    else:
                        continue

                if dim_sizes[dim.name] is None:
                    # We haven't determined the size of this dimension yet,
                    # try to infer it from the data shape.
                    dim_sizes[dim.name] = shape[i]
                else:
                    # We know the dimension size, check if data shape agrees
                    if not dim_sizes[dim.name] <= shape[i]:
                        error('Size of dimension %s was determined to be %d, '
                              'but data for symbol %s has shape %d.'
                              % (dim.name, dim_sizes[dim.name], fname, shape[i]))
                        raise InvalidArgument('Wrong data shape encountered')

        # Make sure we have defined all buffered dimensions and their parents,
        # even if they are not explicitly given or used.
        d_args = [d for d in arguments.values() if isinstance(d, Dimension)]
        for d in d_args:
            if d.is_Buffered:
                if dim_sizes[d.parent.name] is None:
                    dim_sizes[d.parent.name] = dim_sizes[d.name]
                if dim_sizes[d.name] is None:
                    dim_sizes[d.name] = dim_sizes[d.parent.name]

        # Add user-provided block sizes, if any
        dle_arguments = OrderedDict()
        for i in self._dle_state.arguments:
            dim_size = dim_sizes.get(i.original_dim.name, i.original_dim.size)
            if dim_size is None:
                error('Unable to derive size of dimension %s from defaults. '
                      'Please provide an explicit value.' % i.original_dim.name)
                raise InvalidArgument('Unknown dimension size')
            if i.value:
                try:
                    dle_arguments[i.argument.name] = i.value(dim_size)
                except TypeError:
                    dle_arguments[i.argument.name] = i.value
                    # User-provided block size available, do not autotune
                    maybe_autotune = False
            else:
                dle_arguments[i.argument.name] = dim_size
        dim_sizes.update(dle_arguments)

        # Insert loop size arguments from dimension values
        d_args = [d for d in arguments.values() if isinstance(d, Dimension)]
        for d in d_args:
            arguments[d.name] = dim_sizes[d.name]

        # Might have been asked to auto-tune the block size
        if maybe_autotune:
            arguments = self._autotune(arguments)

        # Add profiler structs
        arguments.update(self._extra_arguments())

        # Sanity check argument derivation
        for name, arg in arguments.items():
            if isinstance(arg, SymbolicData) or isinstance(arg, Dimension):
                raise ValueError('Runtime argument %s not defined' % arg)
        return arguments, dim_sizes

    @property
    def ccode(self):
        """Returns the C code generated by this kernel.

        This function generates the internal code block from Iteration
        and Expression objects, and adds the necessary template code
        around it.
        """
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

        return c.Module(header + includes + self._cglobals +
                        elemental_functions + [kernel])

    @property
    def compile(self):
        """
        JIT-compile the Operator using the compiler specified in the global
        configuration dictionary (``configuration['compiler']``).

        It is ensured that JIT compilation will only be performed once per
        :class:`Operator`, reagardless of how many times this method is invoked.

        :returns: The file name of the JIT-compiled function.
        """
        if self._lib is None:
            # No need to recompile if a shared object has already been loaded.
            return jit_compile(self.ccode, configuration['compiler'])
        else:
            return self._lib.name

    @property
    def cfunction(self):
        """Returns the JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:
            basename = self.compile
            self._lib = load(basename, configuration['compiler'])
            self._lib.name = basename

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            argtypes = [c_int if isinstance(v, Dimension) else
                        np.ctypeslib.ndpointer(dtype=v.dtype, flags='C')
                        for v in self.parameters]
            self._cfunction.argtypes = argtypes

        return self._cfunction

    def _arg_data(self, argument):
        return None

    def _arg_shape(self, argument):
        return argument.shape

    def _extra_arguments(self):
        return {}

    def _profile_sections(self, nodes):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        return List(body=nodes)

    def _profile_summary(self, dim_sizes):
        """
        Produce a summary of the performance achieved
        """
        return PerformanceSummary()

    def _autotune(self, arguments):
        """Use auto-tuning on this Operator to determine empirically the
        best block sizes when loop blocking is in use."""
        return arguments

    def _schedule_expressions(self, clusters, ordering):
        """Wrap :class:`Expression` objects, already grouped in :class:`Cluster`
        objects, within nested :class:`Iteration` objects (representing loops),
        according to dimensions and stencils."""

        processed = []
        schedule = OrderedDict()
        for i in clusters:
            # Build the Expression objects to be inserted within an Iteration tree
            expressions = [Expression(v, np.int32 if i.trace.is_index(k) else self.dtype)
                           for k, v in i.trace.items()]

            if not i.stencil.empty:
                root = None
                entries = i.stencil.entries

                # Can I reuse any of the previously scheduled Iterations ?
                index = 0
                for j0, j1 in zip(entries, list(schedule)):
                    if j0 != j1:
                        break
                    root = schedule[j1]
                    index += 1
                needed = entries[index:]

                # Build and insert the required Iterations
                iters = [Iteration([], j.dim, j.dim.size, offsets=j.ofs) for j in needed]
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
                    schedule = OrderedDict(list(schedule.items())[:index] +
                                           list(scheduling.items()))
                    for k, v in list(schedule.items()):
                        schedule[k] = transformer.rebuilt.get(v, v)
            else:
                # No Iterations are needed
                processed.extend(expressions)

        return processed

    def _insert_declarations(self, dle_state, parameters):
        """Populate the Operator's body with the required array and
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

    def _retrieve_loop_ordering(self, expressions):
        """
        Establish a partial ordering for the loops that will appear in the code
        generated by the Operator, based on the order in which dimensions
        appear in the input expressions.
        """
        ordering = []
        for i in flatten(Stencil(i).dimensions for i in expressions):
            if i not in ordering:
                ordering.extend([i, i.parent] if i.is_Buffered else [i])
        return ordering

    def _retrieve_output_fields(self, expressions):
        """Retrieve the fields computed by the Operator."""
        return [i.lhs.base.function for i in expressions if q_indexed(i.lhs)]

    def _retrieve_stencils(self, expressions):
        """Determine the :class:`Stencil` of each provided expression."""
        stencils = [Stencil(i) for i in expressions]
        dimensions = set.union(*[set(i.dimensions) for i in stencils])

        # Filter out aliasing buffered dimensions
        mapper = {d.parent: d for d in dimensions if d.is_Buffered}
        for i in list(stencils):
            for d in i.dimensions:
                if d in mapper:
                    i[mapper[d]] = i.pop(d).union(i.get(mapper[d], set()))

        return stencils

    @property
    def _cparameters(self):
        return super(OperatorBasic, self)._cparameters

    @property
    def _cglobals(self):
        return []


class OperatorForeign(OperatorBasic):
    """
    A special :class:`OperatorBasic` for use outside of Python.
    """

    def arguments(self, *args, **kwargs):
        arguments, _ = super(OperatorForeign, self).arguments(*args, **kwargs)
        return arguments.items()


class OperatorCore(OperatorBasic):
    """
    A special :class:`OperatorBasic` that, besides generation and compilation of
    C code evaluating stencil expressions, can also execute the computation.
    """

    def __init__(self, expressions, **kwargs):
        self.profiler = Profiler()
        super(OperatorCore, self).__init__(expressions, **kwargs)

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply the stencil kernel to a set of data objects"""
        # Build the arguments list to invoke the kernel function
        arguments, dim_sizes = self.arguments(*args, **kwargs)

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

    def _arg_data(self, argument):
        # Ensure we're dealing or deriving numpy arrays
        data = argument.data
        if not isinstance(data, np.ndarray):
            error('No array data found for argument %s' % argument.name)
        return data

    def _arg_shape(self, argument):
        return argument.data.shape

    def _profile_sections(self, nodes):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        mapper = {}
        for node in nodes:
            for itspace in FindSections().visit(node).keys():
                for i in itspace:
                    if IsPerfectIteration().visit(i):
                        # Insert `TimedList` block. This should come from
                        # the profiler, but we do this manually for now.
                        lname = 'loop_%s_%d' % (i.index, len(mapper))
                        mapper[i] = TimedList(gname=self.profiler.varname,
                                              lname=lname, body=i)
                        self.profiler.add(lname)

                        # Estimate computational properties of the timed section
                        # (operational intensity, memory accesses)
                        expressions = FindNodes(Expression).visit(i)
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
            itershape = [i.extent(finish=dim_sizes.get(dims[i].name)) for i in itspace]
            iterspace = reduce(operator.mul, itershape)
            flops = float(profile.ops*iterspace)
            gflops = flops/10**9

            # Compulsory traffic
            datashape = [i.dim.size or dim_sizes[dims[i].name] for i in itspace]
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

    def _extra_arguments(self):
        return OrderedDict([(self.profiler.typename, self.profiler.setup())])

    def _autotune(self, arguments):
        """Use auto-tuning on this Operator to determine empirically the
        best block sizes when loop blocking is in use."""
        if self._dle_state.has_applied_blocking:
            return autotune(self, arguments, self._dle_state.arguments,
                            mode=configuration['autotuning'])
        else:
            return arguments

    @property
    def _cparameters(self):
        cparameters = super(OperatorCore, self)._cparameters
        cparameters += [c.Pointer(c.Value('struct %s' % self.profiler.typename,
                                          self.profiler.varname))]
        return cparameters

    @property
    def _cglobals(self):
        return [self.profiler.ctype, blankline]


class Operator(object):

    def __new__(cls, *args, **kwargs):
        # What type of Operator should I return ?
        cls = OperatorForeign if kwargs.pop('external', False) else OperatorCore

        # Trigger instantiation
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        return obj


# Helpers for performance tracking

PerfEntry = namedtuple('PerfEntry', 'time gflopss oi itershape datashape')
"""A helper to return structured performance data."""


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


# Misc helpers

cnames = {
    'loc_timer': 'loc_timer',
    'glb_timer': 'glb_timer'
}
"""A dict of standard names to be used for code generation."""

Profile = namedtuple('Profile', 'timer ops memory')
"""A helper to track profiled sections of code."""


def set_dle_mode(mode):
    """
    Transform :class:`Operator` input in a format understandable by the DLE.
    """
    if not mode:
        return 'noop', {}
    elif isinstance(mode, str):
        return mode, {}
    elif isinstance(mode, tuple):
        if len(mode) == 1:
            return mode[0], {}
        elif len(mode) == 2 and isinstance(mode[1], dict):
            return mode
    raise TypeError("Illegal DLE mode %s." % str(mode))

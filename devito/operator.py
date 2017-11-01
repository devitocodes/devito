from __future__ import absolute_import

from collections import OrderedDict, namedtuple
from operator import attrgetter

import ctypes
import numpy as np
import sympy

from devito.flow import analyze_iterations
from devito.cgen_utils import Allocator
from devito.compiler import jit_compile, load
from devito.dimension import Dimension
from devito.dle import compose_nodes, filter_iterations, transform
from devito.dse import clusterize, indexify, rewrite, retrieve_terminals
from devito.function import Forward, Backward, CompositeFunction
from devito.types import Object
from devito.logger import bar, error, info
from devito.nodes import Element, Expression, Callable, Iteration, List, LocalExpression
from devito.parameters import configuration
from devito.profiling import create_profile
from devito.stencil import Stencil
from devito.tools import as_tuple, filter_sorted, flatten, numpy_to_ctypes, partial_order
from devito.visitors import (FindScopes, ResolveTimeStepping,
                             SubstituteExpression, Transformer, NestedTransformer)
from devito.exceptions import InvalidArgument, InvalidOperator
from devito.arguments import infer_dimension_values_tuple


class Operator(Callable):

    _default_headers = ['#define _POSIX_C_SOURCE 200809L']
    _default_includes = ['stdlib.h', 'math.h', 'sys/time.h']
    _default_globals = []

    """A special :class:`Callable` to generate and compile C code evaluating
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

        # Header files, etc.
        self._headers = list(self._default_headers)
        self._includes = list(self._default_includes)
        self._globals = list(self._default_globals)

        # Required for compilation
        self._compiler = configuration['compiler']
        self._lib = None
        self._cfunction = None

        # Expression lowering
        expressions = [indexify(s) for s in expressions]
        expressions = [s.xreplace(subs) for s in expressions]

        # Analysis
        self.dtype = self._retrieve_dtype(expressions)
        self.input, self.output, self.dimensions = self._retrieve_symbols(expressions)
        stencils = self._retrieve_stencils(expressions)

        # Extract argument offsets
        self._store_argument_offsets(stencils)

        # Set the direction of time acoording to the given TimeAxis
        for time in [d for d in self.dimensions if d.is_Time]:
            if not time.is_Stepping:
                time.reverse = time_axis == Backward

        # Parameters of the Operator (Dimensions necessary for data casts)
        parameters = self.input + self.dimensions

        # Group expressions based on their Stencil
        clusters = clusterize(expressions, stencils)

        # Apply the Devito Symbolic Engine (DSE) for symbolic optimization
        clusters = rewrite(clusters, mode=set_dse_mode(dse))

        # Wrap expressions with Iterations according to dimensions
        nodes = self._schedule_expressions(clusters)

        # Data dependency analysis. Properties are attached directly to nodes
        nodes = analyze_iterations(nodes)

        # Introduce C-level profiling infrastructure
        nodes, self.profiler = self._profile_sections(nodes, parameters)

        # Resolve and substitute dimensions for loop index variables
        nodes, subs = ResolveTimeStepping().visit(nodes)
        nodes = SubstituteExpression(subs=subs).visit(nodes)

        # Apply the Devito Loop Engine (DLE) for loop optimization
        dle_state = transform(nodes, *set_dle_mode(dle))

        # Update the Operator state based on the DLE
        self.dle_arguments = dle_state.arguments
        self.dle_flags = dle_state.flags
        self.func_table = OrderedDict([(i.name, FunMeta(i, True))
                                       for i in dle_state.elemental_functions])
        parameters.extend([i.argument for i in self.dle_arguments])
        self.dimensions.extend([i.argument for i in self.dle_arguments
                                if isinstance(i.argument, Dimension)])
        self._includes.extend(list(dle_state.includes))

        # Translate into backend-specific representation (e.g., GPU, Yask)
        nodes = self._specialize(dle_state.nodes, parameters)

        # Introduce all required C declarations
        nodes = self._insert_declarations(nodes)

        # Finish instantiation
        super(Operator, self).__init__(self.name, nodes, 'int', parameters, ())

    def arguments(self, **kwargs):
        """ Process any apply-time arguments passed to apply and derive values for
            any remaining arguments
        """
        new_params = {}
        # If we've been passed CompositeFunction objects as kwargs,
        # they might have children that need to be substituted as well.
        for k, v in kwargs.items():
            if isinstance(v, CompositeFunction):
                orig_param_l = [i for i in self.input if i.name == k]
                # If I have been passed a parameter, I must have seen it before
                if len(orig_param_l) == 0:
                    raise InvalidArgument("Parameter %s does not exist in expressions " +
                                          "passed to this Operator" % k)
                # We've made sure the list isn't empty. Names should be unique so it
                # should have exactly one entry
                assert(len(orig_param_l) == 1)
                orig_param = orig_param_l[0]
                # Pull out the children and add them to kwargs
                for orig_child, new_child in zip(orig_param.children, v.children):
                    new_params[orig_child.name] = new_child
        kwargs.update(new_params)

        # Derivation. It must happen in the order [tensors -> dimensions -> scalars]
        for i in self.parameters:
            if i.is_TensorArgument:
                assert(i.verify(kwargs.pop(i.name, None)))
        for d in self.dimensions:
            user_provided_value = kwargs.pop(d.name, None)
            if user_provided_value is not None:
                user_provided_value = infer_dimension_values_tuple(user_provided_value, d.rtargs,
                                                                   self.argument_offsets)
            d.verify(user_provided_value, enforce=True)
        for i in self.parameters:
            if i.is_ScalarArgument:
                user_provided_value = kwargs.pop(i.name, None)
                if user_provided_value is not None:
                    user_provided_value += self.argument_offsets.get(i.name, 0)
                i.verify(user_provided_value, enforce=True)
        dim_sizes = {}
        for d in self.dimensions:
            if d.value is not None:
                _, d_start, d_end = d.value
                # Calculte loop extent
                d_extent = d_end - d_start
            else:
                d_extent = None
            dim_sizes[d.name] = d_extent
        dle_arguments, autotune = self._dle_arguments(dim_sizes)
        dim_sizes.update(dle_arguments)

        autotune = autotune and kwargs.pop('autotune', False)

        # Make sure we've used all arguments passed
        if len(kwargs) > 0:
            raise InvalidArgument("Unknown arguments passed: " + ", ".join(kwargs.keys()))

        mapper = OrderedDict([(d.name, d) for d in self.dimensions])
        for d, v in dim_sizes.items():
            assert(mapper[d].verify(v))

        arguments = self._default_args()

        if autotune:
            arguments = self._autotune(arguments)

        # Clear the temp values we stored in the arg objects since we've pulled them out
        # into the OrderedDict object above
        self._reset_args()

        return arguments, dim_sizes

    def _default_args(self):
        return OrderedDict([(x.name, x.value) for x in self.parameters])

    def _reset_args(self):
        """
        Reset any runtime argument derivation information from a previous run.
        """
        for x in list(self.parameters) + self.dimensions:
            x.reset()

    def _dle_arguments(self, dim_sizes):
        # Add user-provided block sizes, if any
        dle_arguments = OrderedDict()
        autotune = True
        for i in self.dle_arguments:
            dim_size = dim_sizes.get(i.original_dim.name, None)
            if dim_size is None:
                error('Unable to derive size of dimension %s from defaults. '
                      'Please provide an explicit value.' % i.original_dim.name)
                raise InvalidArgument('Unknown dimension size')
            if i.value:
                try:
                    dle_arguments[i.argument.name] = i.value(dim_size)
                except TypeError:
                    dle_arguments[i.argument.name] = i.value
                    autotune = False
            else:
                dle_arguments[i.argument.name] = dim_size
        return dle_arguments, autotune

    @property
    def elemental_functions(self):
        return tuple(i.root for i in self.func_table.values())

    def _store_argument_offsets(self, stencils):
        offs = Stencil.union(*stencils)
        arg_offs = {d: v for d, v in offs.diameter.items()}
        arg_offs.update({d.parent: v for d, v in arg_offs.items() if d.is_Stepping})
        self.argument_offsets = {d.end_name: v for d, v in arg_offs.items()}

    @property
    def compile(self):
        """
        JIT-compile the C code generated by the Operator.

        It is ensured that JIT compilation will only be performed once per
        :class:`Operator`, reagardless of how many times this method is invoked.

        :returns: The file name of the JIT-compiled function.
        """
        if self._lib is None:
            # No need to recompile if a shared object has already been loaded.
            return jit_compile(self.ccode, self._compiler)
        else:
            return self._lib.name

    @property
    def cfunction(self):
        """Returns the JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:
            basename = self.compile
            self._lib = load(basename, self._compiler)
            self._lib.name = basename

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            # Associate a C type to each argument for runtime type check
            argtypes = []
            for i in self.parameters:
                if i.is_ScalarArgument:
                    argtypes.append(numpy_to_ctypes(i.dtype))
                elif i.is_TensorArgument:
                    argtypes.append(np.ctypeslib.ndpointer(dtype=i.dtype, flags='C'))
                else:
                    argtypes.append(ctypes.c_void_p)
            self._cfunction.argtypes = argtypes

        return self._cfunction

    def _profile_sections(self, nodes, parameters):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        return List(body=nodes), None

    def _autotune(self, arguments):
        """Use auto-tuning on this Operator to determine empirically the
        best block sizes when loop blocking is in use."""
        return arguments

    def _schedule_expressions(self, clusters):
        """Create an Iteartion/Expression tree given an iterable of
        :class:`Cluster` objects."""

        # Topologically sort Iterations
        ordering = partial_order([i.stencil.dimensions for i in clusters])
        for i, d in enumerate(list(ordering)):
            if d.is_Stepping:
                ordering.insert(i, d.parent)

        # Build the Iteration/Expression tree
        processed = []
        schedule = OrderedDict()
        atomics = ()
        for i in clusters:
            # Build the Expression objects to be inserted within an Iteration tree
            expressions = [Expression(v, np.int32 if i.trace.is_index(k) else self.dtype)
                           for k, v in i.trace.items()]

            if not i.stencil.empty:
                root = None
                entries = i.stencil.entries

                # Reorder based on the globally-established loop ordering
                entries = sorted(entries, key=lambda i: ordering.index(i.dim))

                # Can I reuse any of the previously scheduled Iterations ?
                index = 0
                for j0, j1 in zip(entries, list(schedule)):
                    if j0 != j1 or j0.dim in atomics:
                        break
                    root = schedule[j1]
                    index += 1
                needed = entries[index:]

                # Build and insert the required Iterations
                iters = [Iteration([], j.dim, j.dim.limits, offsets=j.ofs) for j in
                         needed]
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

            # Track dimensions that cannot be fused at next stage
            atomics = i.atomics

        return List(body=processed)

    def _specialize(self, nodes, parameters):
        """Transform the Iteration/Expression tree into a backend-specific
        representation, such as code to be executed on a GPU or through a
        lower-level tool."""
        return nodes

    def _insert_declarations(self, nodes):
        """Populate the Operator's body with the necessary variable declarations."""

        # Resolve function calls first
        scopes = []
        for k, v in FindScopes().visit(nodes).items():
            if k.is_Call:
                func = self.func_table[k.name]
                if func.local:
                    scopes.extend(FindScopes().visit(func.root, queue=list(v)).items())
            else:
                scopes.append((k, v))

        # Determine all required declarations
        allocator = Allocator()
        mapper = OrderedDict()
        for k, v in scopes:
            if k.is_scalar:
                # Inline declaration
                mapper[k] = LocalExpression(**k.args)
            elif k.write._mem_external:
                # Nothing to do, variable passed as kernel argument
                continue
            elif k.write._mem_stack:
                # On the stack, as established by the DLE
                key = lambda i: not i.is_Parallel
                site = filter_iterations(v, key=key, stop='asap') or [nodes]
                allocator.push_stack(site[-1], k.write)
            else:
                # On the heap, as a tensor that must be globally accessible
                allocator.push_heap(k.write)

        # Introduce declarations on the stack
        for k, v in allocator.onstack:
            mapper[k] = tuple(Element(i) for i in v)
        nodes = NestedTransformer(mapper).visit(nodes)
        for k, v in list(self.func_table.items()):
            if v.local:
                self.func_table[k] = FunMeta(Transformer(mapper).visit(v.root), v.local)

        # Introduce declarations on the heap (if any)
        if allocator.onheap:
            decls, allocs, frees = zip(*allocator.onheap)
            nodes = List(header=decls + allocs, body=nodes, footer=frees)

        return nodes

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

    def _retrieve_stencils(self, expressions):
        """Determine the :class:`Stencil` of each provided expression."""
        stencils = [Stencil(i) for i in expressions]
        dimensions = set.union(*[set(i.dimensions) for i in stencils])

        # Filter out aliasing stepping dimensions
        mapper = {d.parent: d for d in dimensions if d.is_Stepping}
        for i in list(stencils):
            for d in i.dimensions:
                if d in mapper:
                    i[mapper[d]] = i.pop(d).union(i.get(mapper[d], set()))

        return stencils

    def _retrieve_symbols(self, expressions):
        """
        Retrieve the symbolic functions read or written by the Operator,
        as well as all traversed dimensions.
        """
        terms = flatten(retrieve_terminals(i) for i in expressions)

        input = []
        for i in terms:
            try:
                input.append(i.base.function)
            except AttributeError:
                pass
        input = filter_sorted(input, key=attrgetter('name'))

        output = [i.lhs.base.function for i in expressions if i.lhs.is_Indexed]

        indexeds = [i for i in terms if i.is_Indexed]
        dimensions = []
        for indexed in indexeds:
            for i in indexed.indices:
                dimensions.extend([k for k in i.free_symbols
                                   if isinstance(k, Dimension)])
            dimensions.extend(list(indexed.base.function.indices))
        dimensions.extend([d.parent for d in dimensions if d.is_Stepping])
        dimensions = filter_sorted(dimensions, key=attrgetter('name'))

        return input, output, dimensions


class OperatorRunnable(Operator):
    """
    A special :class:`Operator` that, besides generation and compilation of
    C code evaluating stencil expressions, can also execute the computation.
    """

    def __call__(self, **kwargs):
        self.apply(**kwargs)

    def apply(self, **kwargs):
        """Apply the stencil kernel to a set of data objects"""
        # Build the arguments list to invoke the kernel function
        arguments, dim_sizes = self.arguments(**kwargs)

        # Invoke kernel function with args
        self.cfunction(*list(arguments.values()))

        # Output summary of performance achieved
        return self._profile_output(arguments)

    def _profile_output(self, arguments):
        """Return a performance summary of the profiled sections."""
        summary = self.profiler.summary(arguments, self.dtype)
        with bar():
            for k, v in summary.items():
                name = '%s<%s>' % (k, ','.join('%d' % i for i in v.itershape))
                gpointss = ", %.2f GPts/s" % v.gpointss if k == 'main' else ''
                info("Section %s with OI=%.2f computed in %.3f s [%.2f GFlops/s%s]" %
                     (name, v.oi, v.time, v.gflopss, gpointss))
        return summary

    def _profile_sections(self, nodes, parameters):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        nodes, profiler = create_profile(nodes)
        self._globals.append(profiler.cdef)
        parameters.append(Object(profiler.varname, profiler.dtype, profiler.setup()))
        return nodes, profiler


# Misc helpers


FunMeta = namedtuple('FunMeta', 'root local')
"""
Metadata for functions called by an Operator. ``local = True`` means that
the function was generated by Devito itself.
"""


def set_dse_mode(mode):
    """
    Transform :class:`Operator` input in a format understandable by the DLE.
    """
    if not mode:
        return 'noop'
    elif isinstance(mode, str):
        return mode
    else:
        try:
            return ','.join(mode)
        except:
            raise TypeError("Illegal DSE mode %s." % str(mode))


def set_dle_mode(mode):
    """
    Transform :class:`Operator` input in a format understandable by the DLE.
    """
    if not mode:
        return mode, {}
    elif isinstance(mode, str):
        return mode, {}
    elif isinstance(mode, tuple):
        if len(mode) == 1:
            return mode[0], {}
        elif len(mode) == 2 and isinstance(mode[1], dict):
            return mode
    raise TypeError("Illegal DLE mode %s." % str(mode))

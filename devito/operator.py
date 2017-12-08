from __future__ import absolute_import

from collections import OrderedDict, namedtuple
from operator import attrgetter

import ctypes
import numpy as np
import sympy

from devito.cgen_utils import Allocator
from devito.compiler import jit_compile, load
from devito.dimension import Dimension
from devito.dle import transform
from devito.dse import rewrite
from devito.exceptions import InvalidArgument, InvalidOperator
from devito.function import Forward, Backward, CompositeFunction
from devito.logger import bar, error, info
from devito.ir.clusters import clusterize
from devito.ir.iet import (Element, Expression, Callable, Iteration, List,
                           LocalExpression, MapExpressions, ResolveTimeStepping,
                           SubstituteExpression, Transformer, NestedTransformer,
                           analyze_iterations, compose_nodes, filter_iterations)
from devito.ir.support import Stencil
from devito.parameters import configuration
from devito.profiling import create_profile
from devito.symbolics import indexify, retrieve_terminals
from devito.tools import as_tuple, filter_sorted, flatten, numpy_to_ctypes, partial_order
from devito.types import Object
from devito.exceptions import InvalidArgument, InvalidOperator
from devito.arguments import runtime_arguments, ArgumentEngine


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

        # References to local or external routines
        self.func_table = OrderedDict()

        # Expression lowering
        expressions = [indexify(s) for s in expressions]
        expressions = [s.xreplace(subs) for s in expressions]

        # Analysis
        self.dtype = retrieve_dtype(expressions)
        self.input, self.output, self.dimensions = retrieve_symbols(expressions)
        stencils = make_stencils(expressions)

        # Set the direction of time acoording to the given TimeAxis
        for time in [d for d in self.dimensions if d.is_Time]:
            if not time.is_Stepping:
                time.reverse = time_axis == Backward
        
        # Parameters of the Operator (Dimensions necessary for data casts)
        parameters = self.input + self.dimensions

        # Group expressions based on their Stencil and data dependences
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

        # Translate into backend-specific representation (e.g., GPU, Yask)
        nodes = self._specialize(nodes, parameters)

        # Apply the Devito Loop Engine (DLE) for loop optimization
        dle_state = transform(nodes, *set_dle_mode(dle))

        # Update the Operator state based on the DLE
        self.dle_arguments = dle_state.arguments
        self.dle_flags = dle_state.flags
        self.func_table.update(OrderedDict([(i.name, FunMeta(i, True))
                                            for i in dle_state.elemental_functions]))
        parameters.extend([i.argument for i in self.dle_arguments])
        self.dimensions.extend([i.argument for i in self.dle_arguments
                                if isinstance(i.argument, Dimension)])
        self._includes.extend(list(dle_state.includes))

        # Introduce all required C declarations
        nodes = self._insert_declarations(dle_state.nodes)

        # Initialise Argument Engine
        self.argument_engine = ArgumentEngine(stencils, parameters, self.dle_arguments)

        parameters = self.argument_engine.arguments

        # Finish instantiation
        super(Operator, self).__init__(self.name, nodes, 'int', parameters, ())

    def arguments(self, **kwargs):
        """ Process any apply-time arguments passed to apply and derive values for
            any remaining arguments
        """

        arguments, autotune = self.argument_engine.handle(**kwargs)
        
        if autotune:
            arguments = self._autotune(arguments)

        return arguments

    def _default_args(self):
        return OrderedDict([(x.name, x.value) for x in self.parameters])

    @property
    def elemental_functions(self):
        return tuple(i.root for i in self.func_table.values())

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

        # Build the Iteration/Expression tree
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
                    if j0 != j1 or j0.dim in clusters.atomics[i]:
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
        me = MapExpressions()
        for k, v in me.visit(nodes).items():
            if k.is_Call:
                func = self.func_table[k.name]
                if func.local:
                    scopes.extend(me.visit(func.root, queue=list(v)).items())
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
        arguments = self.arguments(**kwargs)

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
        nodes, profiler = create_profile('timers', nodes)
        self._globals.append(profiler.cdef)
        parameters.append(Object(profiler.name, profiler.dtype, profiler.new()))
        return nodes, profiler


# Functions collecting information from a bag of expressions

def retrieve_dtype(expressions):
    """
    Retrieve the data type of a set of expressions. Raise an error if there
    is no common data type (ie, if at least one expression differs in the
    data type).
    """
    lhss = set([s.lhs.base.function.dtype for s in expressions])
    if len(lhss) != 1:
        raise RuntimeError("Expression types mismatch.")
    return lhss.pop()


def retrieve_symbols(expressions):
    """
    Return the :class:`Function` and :class:`Dimension` objects appearing
    in ``expressions``.
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


def make_stencils(expressions):
    """
    Create a :class:`Stencil` for each of the provided expressions. The following
    rules apply: ::

        * A :class:`SteppingDimension` ``d`` is replaced by its parent ``d.parent``.
    """
    stencils = [Stencil(i) for i in expressions]
    dimensions = set.union(*[set(i.dimensions) for i in stencils])

    # Filter out aliasing stepping dimensions
    mapper = {d.parent: d for d in dimensions if d.is_Stepping}
    return [i.replace(mapper) for i in stencils]


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
        if len(mode) == 0:
            return 'noop', {}
        elif isinstance(mode[-1], dict):
            return tuple(flatten(i.split(',') for i in mode[:-1])), mode[-1]
        else:
            return tuple(flatten(i.split(',') for i in mode)), {}
    raise TypeError("Illegal DLE mode %s." % str(mode))

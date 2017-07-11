from __future__ import absolute_import

import operator
from collections import OrderedDict
from ctypes import c_int

import cgen as c
import numpy as np
import sympy

from devito.cgen_utils import Allocator, blankline
from devito.compiler import jit_compile, load
from devito.dimension import time, Dimension
from devito.dle import compose_nodes, filter_iterations, transform
from devito.dse import clusterize, indexify, rewrite, q_indexed
from devito.interfaces import Forward, Backward, CompositeData, SymbolicData
from devito.logger import bar, error, info
from devito.nodes import Element, Expression, Function, Iteration, List, LocalExpression
from devito.parameters import configuration
from devito.profiling import Profiler, create_profile
from devito.stencil import Stencil
from devito.tools import as_tuple, filter_ordered, flatten
from devito.visitors import (FindSymbols, FindScopes, ResolveIterationVariable,
                             SubstituteExpression, Transformer, NestedTransformer)
from devito.exceptions import InvalidArgument, InvalidOperator
from devito.arguments import ScalarArgument, log_args, ArgumentProvider


class Operator(Function):

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
        clusters = rewrite(clusters, mode=set_dse_mode(dse))

        # Wrap expressions with Iterations according to dimensions
        nodes = self._schedule_expressions(clusters, ordering)

        # Introduce C-level profiling infrastructure
        nodes, self.profiler = self._profile_sections(nodes)

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
        nodes, elemental_functions = dle_state.trees
        self.dle_arguments = dle_state.arguments
        self.dle_flags = dle_state.flags

        # Update the Operator arguments
        parameters += [i.argument for i in self.dle_arguments]
        self._includes.extend(list(dle_state.includes))

        self.symbolic_data = [i for i in parameters if isinstance(i, SymbolicData)]
        dims = [i for i in parameters if isinstance(i, Dimension)]
        d_parents = [d.parent for d in dims if hasattr(d, 'parent')]

        self.dims = list(set(dims + d_parents))

        assert(all(isinstance(i, ArgumentProvider) for i in self.symbolic_data +
                   self.dims))

        parameters = list(set(parameters + d_parents))

        # Translate into backend-specific representation (e.g., GPU, Yask)
        nodes, elemental_functions = self._specialize(nodes, elemental_functions)

        # Introduce all required C declarations
        nodes, elemental_functions = self._insert_declarations(nodes,
                                                               elemental_functions,
                                                               dle_state.func_table,
                                                               parameters)

        # Finish instantiation
        super(Operator, self).__init__(self.name, nodes, 'int', parameters, ())
        self.elemental_functions = elemental_functions

    def _reset_args(self):
        """ Reset any runtime argument derivation information from a previous run
        """
        for x in self.tensor_args + self.dims + self.scalar_args:
            x.reset()

    def arguments(self, *args, **kwargs):
        """ Process any apply-time arguments passed to apply and derive values for
            any remaining arguments
        """
        new_params = {}
        # If we've been passed CompositeData objects as kwargs, they might have children
        # that need to be substituted as well.
        for k, v in kwargs.items():
            if isinstance(v, CompositeData):
                orig_param_l = [i for i in self.symbolic_data if i.name == k]
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

        for ta in self.tensor_args:
            assert(ta.verify(kwargs.pop(ta.name, None)))

        for d in self.dims:
            d.verify(kwargs.pop(d.name, None))

        for s in self.scalar_args:
            s.verify(kwargs.pop(s.name, None))

        dim_sizes = OrderedDict([(d.name, d.value) for d in self.dims])
        dim_names = OrderedDict([(d.name, d) for d in self.dims])
        dle_arguments, autotune = self._dle_arguments(dim_sizes)
        dim_sizes.update(dle_arguments)

        autotune = autotune and kwargs.pop('autotune', False)

        # Make sure we've used all arguments passed
        if len(kwargs) > 0:
            raise InvalidArgument("Unknown arguments passed: " + ", ".join(kwargs.keys()))

        for d, v in dim_sizes.items():
            assert(dim_names[d].verify(v))

        arguments = self._default_args()

        arguments.update(self._extra_arguments())

        if autotune:
            arguments = self._autotune(arguments)

        # Clear the temp values we stored in the arg objects since we've pulled them out
        # into the OrderedDict object above
        self._reset_args()

        log_args(arguments)

        return arguments, dim_sizes

    def _default_args(self):
        return OrderedDict([(x.name, x.value) for x in self.parameters])

    def _dle_arguments(self, dim_sizes):
        # Add user-provided block sizes, if any
        dle_arguments = OrderedDict()
        auto_tune = True
        for i in self.dle_arguments:
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
                    auto_tune = False
            else:
                dle_arguments[i.argument.name] = dim_size
        return dle_arguments, auto_tune

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
            argtypes = [c_int if isinstance(v, ScalarArgument) else
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
        return List(body=nodes), Profiler()

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

        return List(body=processed)

    def _specialize(self, nodes, elemental_functions):
        """Transform the Iteration/Expression tree into a backend-specific
        representation, such as code to be executed on a GPU or through a
        lower-level tool."""
        return nodes, elemental_functions

    def _insert_declarations(self, nodes, elemental_functions, func_table, parameters):
        """Populate the Operator's body with the required array and variable
        declarations, to generate a legal C file."""

        # Resolve function calls first
        scopes = []
        for k, v in FindScopes().visit(nodes).items():
            if k.is_FunCall:
                function = func_table[k.name]
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
                key = lambda i: not i.is_Parallel
                site = filter_iterations(v, key=key, stop='asap') or [nodes]
                allocator.push_stack(site[-1], k.output_function)
            else:
                # On the heap, as a tensor that must be globally accessible
                allocator.push_heap(k.output_function)

        # Introduce declarations on the stack
        for k, v in allocator.onstack:
            mapper[k] = tuple(Element(i) for i in v)
        nodes = NestedTransformer(mapper).visit(nodes)
        elemental_functions = Transformer(mapper).visit(elemental_functions)

        # TODO
        # Use x_block_size+2, y_block_size+2 instead of x_size, y_size as shape of
        # the tensor functions

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
        return super(Operator, self)._cparameters

    @property
    def _cglobals(self):
        return []


class OperatorRunnable(Operator):
    """
    A special :class:`Operator` that, besides generation and compilation of
    C code evaluating stencil expressions, can also execute the computation.
    """

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply the stencil kernel to a set of data objects"""
        # Build the arguments list to invoke the kernel function
        arguments, dim_sizes = self.arguments(*args, **kwargs)

        # Invoke kernel function with args
        self.cfunction(*list(arguments.values()))

        # Output summary of performance achieved
        summary = self.profiler.summary(dim_sizes, self.dtype)
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
        return create_profile(nodes)

    def _extra_arguments(self):
        return OrderedDict([(self.profiler.typename, self.profiler.setup())])

    @property
    def _cparameters(self):
        cparameters = super(OperatorRunnable, self)._cparameters
        cparameters += [c.Pointer(c.Value('struct %s' % self.profiler.typename,
                                          self.profiler.varname))]
        return cparameters

    @property
    def _cglobals(self):
        return [self.profiler.ctype, blankline]


# Misc helpers


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
        return 'noop', {}
    elif isinstance(mode, str):
        return mode, {}
    elif isinstance(mode, tuple):
        if len(mode) == 1:
            return mode[0], {}
        elif len(mode) == 2 and isinstance(mode[1], dict):
            return mode
    raise TypeError("Illegal DLE mode %s." % str(mode))

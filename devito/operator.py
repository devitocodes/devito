from __future__ import absolute_import

from collections import OrderedDict

import ctypes
import numpy as np
import sympy

from devito.arguments import ArgumentMap
from devito.compiler import jit_compile, load
from devito.dimension import Dimension
from devito.dle import transform
from devito.dse import rewrite
from devito.exceptions import InvalidOperator
from devito.logger import bar, info
from devito.ir.equations import LoweredEq
from devito.ir.clusters import clusterize
from devito.ir.iet import (Callable, List, MetaCall, iet_build, iet_insert_C_decls,
                           ArrayCast, PointerCast, derive_parameters)
from devito.parameters import configuration
from devito.profiling import create_profile
from devito.tools import as_tuple, flatten, filter_sorted, numpy_to_ctypes
from devito.types import Object


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

        # Expression lowering and analysis
        expressions = [LoweredEq(e, subs=subs) for e in expressions]
        self.dtype = retrieve_dtype(expressions)
        self.input = filter_sorted(flatten(e.reads for e in expressions))
        self.output = filter_sorted(flatten(e.writes for e in expressions))
        self.dimensions = filter_sorted(flatten(e.dimensions for e in expressions))

        # Group expressions based on their iteration space and data dependences,
        # and apply the Devito Symbolic Engine (DSE) for flop optimization
        clusters = clusterize(expressions)
        clusters = rewrite(clusters, mode=set_dse_mode(dse))

        # Lower Clusters to an Iteration/Expression tree (IET)
        nodes = iet_build(clusters, self.dtype)

        # Introduce C-level profiling infrastructure
        nodes, self.profiler = self._profile_sections(nodes)

        # Translate into backend-specific representation (e.g., GPU, Yask)
        nodes = self._specialize(nodes)

        # Apply the Devito Loop Engine (DLE) for loop optimization
        dle_state = transform(nodes, *set_dle_mode(dle))

        # Update the Operator state based on the DLE
        self.dle_arguments = dle_state.arguments
        self.dle_flags = dle_state.flags
        self.func_table.update(OrderedDict([(i.name, MetaCall(i, True))
                                            for i in dle_state.elemental_functions]))
        self.dimensions.extend([i.argument for i in self.dle_arguments
                                if isinstance(i.argument, Dimension)])
        self._includes.extend(list(dle_state.includes))

        # Introduce the required symbol declarations
        nodes = iet_insert_C_decls(dle_state.nodes, self.func_table)

        # Insert data and pointer casts for array parameters and profiling structs
        nodes = self._build_casts(nodes)

        # Derive parameters as symbols not defined in the kernel itself
        parameters = self._build_parameters(nodes)

        # Finish instantiation
        super(Operator, self).__init__(self.name, nodes, 'int', parameters, ())

    def _argument_defaults(self, arguments):
        """
        Derive all default values from parameters and ensure uniqueness.
        """
        default_args = ArgumentMap()
        for p in self.input:
            if p.name not in arguments:
                default_args.update(p.argument_defaults())
        for p in self.dimensions:
            if p.name not in arguments and p.is_Sub:
                default_args.update(p.argument_defaults(default_args))
        return {k: default_args.reduce(k) for k in default_args if k not in arguments}

    def arguments(self, **kwargs):
        """
        Process runtime arguments passed to ``.apply()` and derive
        default values for any remaining arguments.
        """
        arguments = {}
        # First, we insert user-provided override
        for p in self.input + self.dimensions:
            arguments.update(p.argument_values(**kwargs))

        # Second, derive all remaining default values from parameters
        arguments.update(self._argument_defaults(arguments))

        # Derive additional values for DLE arguments
        # TODO: This is not pretty, but it works for now. Ideally, the
        # DLE arguments would be massaged into the IET so as to comply
        # with the rest of the argument derivation procedure.
        for arg in self.dle_arguments:
            dim = arg.argument
            osize = arguments[arg.original_dim.symbolic_size.name]
            if dim.symbolic_size in self.parameters:
                if arg.value is None:
                    arguments[dim.symbolic_size.name] = osize
                elif isinstance(arg.value, int):
                    arguments[dim.symbolic_size.name] = arg.value
                else:
                    arguments[dim.symbolic_size.name] = arg.value(osize)

        # Add in the profiler argument
        arguments[self.profiler.name] = self.profiler.new()

        # Execute autotuning and adjust arguments accordingly
        autotune = kwargs.pop('autotune', False)
        if autotune:
            # AT assumes and ordered dict, so let's feed it one
            at_args = OrderedDict([(p.name, arguments[p.name]) for p in self.parameters])
            arguments = self._autotune(at_args)

        # Check all argument are present
        for p in self.parameters:
            if p.name not in arguments:
                raise ValueError("No value found for parameter %s" % p.name)

        return arguments

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
                if i.is_Object:
                    argtypes.append(ctypes.c_void_p)
                elif i.is_Scalar:
                    argtypes.append(numpy_to_ctypes(i.dtype))
                elif i.is_Tensor:
                    argtypes.append(np.ctypeslib.ndpointer(dtype=i.dtype, flags='C'))
                else:
                    argtypes.append(ctypes.c_void_p)
            self._cfunction.argtypes = argtypes

        return self._cfunction

    def _profile_sections(self, nodes):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        return List(body=nodes), None

    def _autotune(self, arguments):
        """Use auto-tuning on this Operator to determine empirically the
        best block sizes when loop blocking is in use."""
        return arguments

    def _specialize(self, nodes):
        """Transform the Iteration/Expression tree into a backend-specific
        representation, such as code to be executed on a GPU or through a
        lower-level tool."""
        return nodes

    def _build_parameters(self, nodes):
        """Determine the Operator parameters based on the Iteration/Expression
        tree ``nodes``."""
        return derive_parameters(nodes, True)

    def _build_casts(self, nodes):
        """Introduce array and pointer casts at the top of the Iteration/Expression
        tree ``nodes``."""
        casts = [ArrayCast(f) for f in self.input if f.is_Tensor and f._mem_external]
        profiler = Object(self.profiler.name, self.profiler.dtype, self.profiler.new)
        casts.append(PointerCast(profiler))
        return List(body=casts + [nodes])


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
        arg_values = [arguments[p.name] for p in self.parameters]
        self.cfunction(*arg_values)

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

    def _profile_sections(self, nodes,):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        nodes, profiler = create_profile('timers', nodes)
        self._globals.append(profiler.cdef)
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

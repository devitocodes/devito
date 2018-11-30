from collections import OrderedDict
from functools import reduce
from operator import mul

from cached_property import cached_property
import ctypes
import numpy as np
import sympy

from devito.compiler import jit_compile, load, save
from devito.dle import transform
from devito.dse import rewrite
from devito.exceptions import InvalidOperator
from devito.logger import info, perf, warning
from devito.ir.equations import LoweredEq
from devito.ir.clusters import clusterize
from devito.ir.iet import (Callable, List, MetaCall, iet_build, iet_insert_C_decls,
                           ArrayCast, derive_parameters)
from devito.ir.stree import st_build
from devito.parameters import configuration
from devito.profiling import create_profile
from devito.symbolics import indexify
from devito.tools import (Signer, ReducerMap, as_tuple, flatten, filter_sorted,
                          numpy_to_ctypes, split)


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

        # Header files, etc.
        self._headers = list(self._default_headers)
        self._includes = list(self._default_includes)
        self._globals = list(self._default_globals)

        # Required for compilation
        self._compiler = configuration['compiler']
        self._lib = None
        self._cfunction = None

        # References to local or external routines
        self._func_table = OrderedDict()

        # Internal state. May be used to store information about previous runs,
        # autotuning reports, etc
        self._state = {}

        # Expression lowering: indexification, substitution rules, specialization
        expressions = [indexify(i) for i in expressions]
        expressions = self._apply_substitutions(expressions, subs)
        expressions = self._specialize_exprs(expressions)

        # Expression analysis
        self.input = filter_sorted(flatten(e.reads for e in expressions))
        self.output = filter_sorted(flatten(e.writes for e in expressions))
        self.dimensions = filter_sorted(flatten(e.dimensions for e in expressions))

        # Group expressions based on their iteration space and data dependences,
        # and apply the Devito Symbolic Engine (DSE) for flop optimization
        clusters = clusterize(expressions)
        clusters = rewrite(clusters, mode=set_dse_mode(dse))
        self._dtype, self._dspace = clusters.meta

        # Lower Clusters to a Schedule tree
        stree = st_build(clusters)

        # Lower Schedule tree to an Iteration/Expression tree (IET)
        iet = iet_build(stree)

        # Insert code for C-level performance profiling
        iet, self._profiler = self._profile_sections(iet)

        # Translate into backend-specific representation
        iet = self._specialize_iet(iet, **kwargs)

        # Insert the required symbol declarations
        iet = iet_insert_C_decls(iet, self._func_table)

        # Insert code for MPI support
        iet = self._generate_mpi(iet, **kwargs)

        # Insert data and pointer casts for array parameters
        iet = self._build_casts(iet)

        # Derive parameters as symbols not defined in the kernel itself
        parameters = self._build_parameters(iet)

        # Finish instantiation
        super(Operator, self).__init__(self.name, iet, 'int', parameters, ())

    def _prepare_arguments(self, **kwargs):
        """
        Process runtime arguments passed to ``.apply()` and derive
        default values for any remaining arguments.
        """
        # Process data-carriers (first overrides, then fill up with whatever is needed)
        args = ReducerMap()
        args.update([p._arg_values(**kwargs) for p in self.input if p.name in kwargs])
        args.update([p._arg_values() for p in self.input if p.name not in args])
        args = args.reduce_all()

        # All TensorFunctions should be defined on the same Grid
        functions = [kwargs.get(p, p) for p in self.input if p.is_TensorFunction]
        mapper = ReducerMap([('grid', i.grid) for i in functions if i.grid])
        try:
            grid = mapper.unique('grid')
        except (KeyError, ValueError):
            if mapper and configuration['mpi']:
                raise RuntimeError("Multiple `Grid`s found before `apply`")
            grid = None

        # Process dimensions (derived go after as they might need/affect their parents)
        derived, main = split(self.dimensions, lambda i: i.is_Derived)
        for p in main:
            args.update(p._arg_values(args, self._dspace[p], grid, **kwargs))
        for p in derived:
            args.update(p._arg_values(args, self._dspace[p], grid, **kwargs))

        # Sanity check
        for p in self.input:
            p._arg_check(args, self._dspace[p])

        # Add in the profiler argument
        args[self._profiler.name] = self._profiler.timer.reset()

        # Add in any backend-specific argument
        args.update(kwargs.pop('backend', {}))

        # Execute autotuning and adjust arguments accordingly
        args = self._autotune(args, kwargs.pop('autotune', configuration['autotuning']))

        # Check all user-provided keywords are known to the Operator
        if not configuration['ignore-unknowns']:
            for k, v in kwargs.items():
                if k not in self._known_arguments:
                    raise ValueError("Unrecognized argument %s=%s" % (k, v))

        return args

    def _postprocess_arguments(self, args, **kwargs):
        """
        Process runtime arguments upon returning from ``.apply()``.
        """
        for p in self.output:
            p._arg_apply(args[p.name], kwargs.get(p.name))

    @cached_property
    def _known_arguments(self):
        """Return an iterable of arguments that can be passed to ``apply``
        when running the operator."""
        ret = set.union(*[set(i._arg_names) for i in self.input + self.dimensions])
        return tuple(sorted(ret))

    def arguments(self, **kwargs):
        args = self._prepare_arguments(**kwargs)
        # Check all arguments are present
        for p in self.parameters:
            if args.get(p.name) is None:
                raise ValueError("No value found for parameter %s" % p.name)
        return args

    @property
    def _efuncs(self):
        return tuple(i.root for i in self._func_table.values())

    @cached_property
    def _soname(self):
        """
        A unique name for the shared object resulting from the jit-compilation
        of this Operator.
        """
        return Signer._digest(self, configuration)

    def _compile(self):
        """
        JIT-compile the C code generated by the Operator.

        It is ensured that JIT compilation will only be performed once per
        :class:`Operator`, reagardless of how many times this method is invoked.

        :returns: The file name of the JIT-compiled function.
        """
        if self._lib is None:
            jit_compile(self._soname, str(self.ccode), self._compiler)

    @property
    def cfunction(self):
        """Returns the JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:
            self._compile()
            self._lib = load(self._soname)
            self._lib.name = self._soname

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            # Associate a C type to each argument for runtime type check
            argtypes = []
            for i in self.parameters:
                if i.is_Object:
                    argtypes.append(i.dtype)
                elif i.is_Scalar:
                    argtypes.append(numpy_to_ctypes(i.dtype))
                elif i.is_Tensor:
                    argtypes.append(np.ctypeslib.ndpointer(dtype=i.dtype, flags='C'))
                else:
                    argtypes.append(ctypes.c_void_p)
            self._cfunction.argtypes = argtypes

        return self._cfunction

    def _profile_sections(self, iet):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        return List(body=iet), None

    def _autotune(self, args, setup):
        """Use auto-tuning on this Operator to determine empirically the
        best block sizes when loop blocking is in use."""
        return args

    def _apply_substitutions(self, expressions, subs):
        """
        Transform ``expressions`` by: ::

            * Applying any user-provided symbolic substitution;
            * Replacing :class:`Dimension`s with :class:`SubDimension`s based
              on the expression :class:`SubDomain`.
        """
        processed = []
        for e in expressions:
            mapper = subs.copy()
            if e.subdomain:
                mapper.update(e.subdomain.dimension_map)
            processed.append(e.xreplace(mapper))
        return processed

    def _specialize_exprs(self, expressions):
        """Transform ``expressions`` into a backend-specific representation."""
        return [LoweredEq(i) for i in expressions]

    def _specialize_iet(self, iet, **kwargs):
        """Transform the Iteration/Expression tree into a backend-specific
        representation, such as code to be executed on a GPU or through a
        lower-level tool."""
        dle = kwargs.get("dle", configuration['dle'])

        # Apply the Devito Loop Engine (DLE) for loop optimization
        state = transform(iet, *set_dle_mode(dle))

        self._func_table.update(OrderedDict([(i.name, MetaCall(i, True))
                                             for i in state.efuncs]))
        self.dimensions.extend(state.dimensions)
        self.input.extend(state.input)
        self._includes.extend(state.includes)

        return state.nodes

    def _generate_mpi(self, iet, **kwargs):
        """Transform the Iteration/Expression tree adding nodes performing halo
        exchanges right before :class:`Iteration`s accessing distributed
        :class:`TensorFunction`s."""
        return iet

    def _build_parameters(self, iet):
        """Determine the Operator parameters based on the Iteration/Expression
        tree ``iet``."""
        return derive_parameters(iet, True)

    def _build_casts(self, iet):
        """Introduce array and pointer casts at the top of the Iteration/Expression
        tree ``iet``."""
        casts = [ArrayCast(f) for f in self.input if f.is_Tensor and f._mem_external]
        return List(body=casts + [iet])

    @cached_property
    def _mem_summary(self):
        """The amount of data, in bytes, used by the Operator. This is provided as
        symbolic expressions, one symbolic expression for each memory scope (external,
        stack, heap)."""
        tensors = [i for i in derive_parameters(self) if i.is_Tensor]

        summary = {}

        external = [i.symbolic_shape for i in tensors if i._mem_external]
        external = sum(reduce(mul, i, 1) for i in external)*self._dtype().itemsize
        summary['external'] = external

        heap = [i.symbolic_shape for i in tensors if i._mem_heap]
        heap = sum(reduce(mul, i, 1) for i in heap)*self._dtype().itemsize
        summary['heap'] = heap

        stack = [i.symbolic_shape for i in tensors if i._mem_stack]
        stack = sum(reduce(mul, i, 1) for i in stack)*self._dtype().itemsize
        summary['stack'] = stack

        summary['total'] = external + heap + stack

        return summary

    def __getstate__(self):
        if self._lib:
            state = dict(self.__dict__)
            state.pop('_soname')
            # The compiled shared-object will be pickled; upon unpickling, it
            # will be restored into a potentially different temporary directory,
            # so the entire process during which the shared-object is loaded and
            # given to ctypes must be performed again
            state['_lib'] = None
            state['_cfunction'] = None
            with open(self._lib._name, 'rb') as f:
                state['binary'] = f.read()
            return state
        else:
            return self.__dict__

    def __setstate__(self, state):
        soname = state.pop('_soname', None)
        binary = state.pop('binary', None)
        for k, v in state.items():
            setattr(self, k, v)
        # If the `sonames` don't match, there *might* be a hidden bug as the
        # unpickled Operator might be generating code that differs from that
        # generated by the pickled Operator. For example, a stupid bug that we
        # had to fix was due to rebuilding SymPy expressions which weren't
        # automatically getting the flag `evaluate=False`, thus producing x+2
        # on the unpickler instead of x+1+1).  However, different `sonames`
        # doesn't necessarily means there's a bug: if the unpickler and the
        # pickler are two distinct processes and the unpickler runs with a
        # different `configuration` dictionary, then the `sonames` might indeed
        # be different, depending on which entries in `configuration` differ.
        if soname is not None:
            if soname != self._soname:
                warning("The pickled and unpickled Operators have different .sonames; "
                        "this might be a bug, or simply a harmless difference in "
                        "`configuration`. You may check they produce the same code.")
            save(self._soname, binary, self._compiler)


class OperatorRunnable(Operator):
    """
    A special :class:`Operator` that, besides generation and compilation of
    C code evaluating stencil expressions, can also execute the computation.
    """

    def __call__(self, **kwargs):
        self.apply(**kwargs)

    def apply(self, **kwargs):
        """
        Run the operator.

        Without additional parameters specified, the operator runs on the same
        data objects used to build it -- the so called ``default arguments``.

        Optionally, any of the operator default arguments may be replaced by
        passing suitable key-value parameters. Given ``apply(k=v, ...)``,
        ``(k, v)`` may be used to: ::

            * replace a constant (scalar) used by the operator. In this case,
                ``k`` is the name of the constant; ``v`` is either an object
                of type :class:`Constant` or an actual scalar value.
            * replace a function (tensor) used by the operator. In this case,
                ``k`` is the name of the function; ``v`` is either an object
                of type :class:`TensorFunction` or a :class:`numpy.ndarray`.
            * alter the iteration interval along a given :class:`Dimension`
                ``d``, which represents a subset of the operator iteration space.
                By default, the operator runs over all iterations within the
                compact interval ``[d_m, d_M]``, in which ``d_m`` and ``d_M``
                are, respectively, the smallest and largest integers not causing
                out-of-bounds memory accesses. In this case, ``k`` can be any
                of ``(d_m, d_M, d_n)``; ``d_n`` can be used to indicate to run
                for exactly ``n`` iterations starting at ``d_m``. ``d_n`` is
                ignored (raising a warning) if ``d_M`` is also provided. ``v`` is
                an integer value.

        Examples
        --------
        The following operator implements a trivial time-marching method which
        adds 1 to every grid point at every time iteration.

        >>> from devito import Eq, Grid, TimeFunction, Operator
        >>> grid = Grid(shape=(3, 3))
        >>> u = TimeFunction(name='u', grid=grid, save=3)
        >>> op = Operator(Eq(u.forward, u + 1))

        The operator is run by calling

        >>> op.apply()

        As no key-value parameters are specified, the operator runs with its
        default arguments, namely ``u=u, x_m=0, x_M=2, y_m=0, y_M=2, time_m=0,
        time_M=1``. Note that one can access the operator dimensions via the
        ``grid`` object (e.g., ``grid.dimensions`` for the ``x`` and ``y``
        space dimensions).

        At this point, the same operator can be used for a completely different
        run, for example

        >>> u2 = TimeFunction(name='u', grid=grid, save=5)
        >>> op.apply(u=u2, x_m=1, y_M=1)

        Now, the operator will run with a different set of arguments, namely
        ``u=u2, x_m=1, x_M=2, y_m=0, y_M=1, time_m=0, time_M=3``.

        To run an operator that only uses buffered :class:`TimeFunction`s,
        the maximum iteration point along the time dimension must be explicitly
        specified (otherwise, the operator wouldn't know how many iterations
        to run).

        >>> u3 = TimeFunction(name='u', grid=grid)
        >>> op = Operator(Eq(u3.forward, u3 + 1))
        >>> op.apply(time_M=10)
        """
        # Build the arguments list to invoke the kernel function
        args = self.arguments(**kwargs)

        # Invoke kernel function with args
        arg_values = [args[p.name] for p in self.parameters]
        try:
            self.cfunction(*arg_values)
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
        return self._profile_output(args)

    def _profile_output(self, args):
        """Return a performance summary of the profiled sections."""
        summary = self._profiler.summary(args, self._dtype)
        info("Operator `%s` run in %.2f s" % (self.name, sum(summary.timings.values())))
        for k, v in summary.items():
            itershapes = [",".join(str(i) for i in its) for its in v.itershapes]
            if len(itershapes) > 1:
                name = "%s<%s>" % (k, ",".join("<%s>" % i for i in itershapes))
            elif len(itershapes) == 1:
                name = "%s<%s>" % (k, itershapes[0])
            else:
                name = None
            gpointss = ", %.2f GPts/s" % v.gpointss if v.gpointss else ''
            perf("* %s with OI=%.2f computed in %.3f s [%.2f GFlops/s%s]" %
                 (name, v.oi, v.time, v.gflopss, gpointss))
        return summary

    def _profile_sections(self, iet):
        """Instrument the Iteration/Expression tree for C-level profiling."""
        profiler = create_profile('timers')
        iet = profiler.instrument(iet)
        self._globals.append(profiler.cdef)
        self._includes.extend(profiler._default_includes)
        self._func_table.update({i: MetaCall(None, False) for i in profiler._ext_calls})
        return iet, profiler


# Misc helpers


def set_dse_mode(mode):
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
    if not mode:
        return mode, {}
    elif isinstance(mode, str):
        return mode, {}
    elif isinstance(mode, tuple):
        if len(mode) == 0:
            return 'noop', {}
        elif isinstance(mode[-1], dict):
            if len(mode) == 2:
                return mode
            else:
                return tuple(flatten(i.split(',') for i in mode[:-1])), mode[-1]
        else:
            return tuple(flatten(i.split(',') for i in mode)), {}
    raise TypeError("Illegal DLE mode %s." % str(mode))

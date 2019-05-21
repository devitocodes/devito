from collections import OrderedDict
from functools import reduce
from operator import mul
from math import ceil

from cached_property import cached_property
import ctypes

from devito.dle import transform
from devito.dse import rewrite
from devito.equation import Eq
from devito.exceptions import InvalidOperator
from devito.logger import info, perf, warning
from devito.ir.equations import LoweredEq
from devito.ir.clusters import clusterize
from devito.ir.iet import (Callable, MetaCall, iet_build, iet_insert_decls,
                           iet_insert_casts, derive_parameters)
from devito.ir.stree import st_build
from devito.mpi import MPI
from devito.parameters import configuration
from devito.profiling import create_profile
from devito.symbolics import indexify
from devito.tools import (DAG, Signer, ReducerMap, as_tuple, flatten, filter_ordered,
                          filter_sorted, split)
from devito.types import Dimension

__all__ = ['Operator']


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
        * dse : str
            Aggressiveness of the Devito Symbolic Engine for flop
            optimization. Defaults to ``configuration['dse']``.
        * dle : str
            Aggressiveness of the Devito Loop Engine for loop-level
            optimization. Defaults to ``configuration['dle']``.

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

    _default_headers = ['#define _POSIX_C_SOURCE 200809L']
    _default_includes = ['stdlib.h', 'math.h', 'sys/time.h']
    _default_globals = []

    def __init__(self, expressions, **kwargs):
        expressions = as_tuple(expressions)

        # Input check
        if any(not isinstance(i, Eq) for i in expressions):
            raise InvalidOperator("Only `devito.Eq` expressions are allowed.")

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
        self._state = self._initialize_state(**kwargs)

        # Form and gather any required implicit expressions
        expressions = self._add_implicit(expressions)

        # Expression lowering: evaluation of derivatives, indexification,
        # substitution rules, specialization
        expressions = [i.evaluate for i in expressions]
        # split vector equation to list of equation
        expressions = [j for i in expressions for j in i._flatten]
        expressions = [indexify(i) for i in expressions]
        expressions = self._apply_substitutions(expressions, subs)
        expressions = self._specialize_exprs(expressions)

        # Expression analysis
        self._input = filter_sorted(flatten(e.reads + e.writes for e in expressions))
        self._output = filter_sorted(flatten(e.writes for e in expressions))
        self._dimensions = filter_sorted(flatten(e.dimensions for e in expressions))

        # Group expressions based on their iteration space and data dependences,
        # and apply the Devito Symbolic Engine (DSE) for flop optimization
        clusters = clusterize(expressions)
        clusters = rewrite(clusters, mode=set_dse_mode(dse))
        self._dtype, self._dspace = clusters.meta

        # Lower Clusters to a Schedule tree
        stree = st_build(clusters)

        # Lower Schedule tree to an Iteration/Expression tree (IET)
        iet = iet_build(stree)
        iet, self._profiler = self._profile_sections(iet)
        iet = self._specialize_iet(iet, **kwargs)

        # Derive all Operator parameters based on the IET
        parameters = derive_parameters(iet, True)

        # Finalization: introduce declarations, type casts, etc
        iet = self._finalize(iet, parameters)

        super(Operator, self).__init__(self.name, iet, 'int', parameters, ())

    # Read-only fields exposed to the outside world

    def __call__(self, **kwargs):
        self.apply(**kwargs)

    @cached_property
    def output(self):
        return tuple(self._output)

    @cached_property
    def dimensions(self):
        return tuple(self._dimensions)

    @cached_property
    def input(self):
        ret = [i for i in self._input + list(self.parameters) if i.is_Input]
        return tuple(filter_ordered(ret))

    @cached_property
    def objects(self):
        return tuple(i for i in self.parameters if i.is_Object)

    # Compilation

    def _initialize_state(self, **kwargs):
        return {'optimizations': {k: kwargs.get(k, configuration[k])
                                  for k in ('dse', 'dle')}}

    def _add_implicit(self, expressions):
        """
        Create and add any associated implicit expressions.

        Implicit expressions are those not explicitly defined by the user
        but instead are requisites of some specified functionality.
        """
        processed = []
        seen = set()
        for e in expressions:
            if e.subdomain:
                try:
                    dims = [d.root for d in e.free_symbols if isinstance(d, Dimension)]
                    sub_dims = [d.root for d in e.subdomain.dimensions]
                    dims = [d for d in dims if d not in frozenset(sub_dims)]
                    dims.append(e.subdomain.implicit_dimension)
                    if e.subdomain not in seen:
                        processed.extend([i.func(*i.args, implicit_dims=dims) for i in
                                          e.subdomain._create_implicit_exprs()])
                        seen.add(e.subdomain)
                    dims.extend(e.subdomain.dimensions)
                    new_e = Eq(e.lhs, e.rhs, subdomain=e.subdomain, implicit_dims=dims)
                    processed.append(new_e)
                except AttributeError:
                    processed.append(e)
            else:
                processed.append(e)
        return processed

    def _apply_substitutions(self, expressions, subs):
        """
        Transform ``expressions`` by: ::

            * Applying any user-provided symbolic substitution;
            * Replacing Dimensions with SubDimensions based on expression SubDomains.
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

    def _profile_sections(self, iet):
        """Instrument the IET for C-level profiling."""
        profiler = create_profile('timers')
        iet = profiler.instrument(iet)
        self._includes.extend(profiler._default_includes)
        self._func_table.update({i: MetaCall(None, False) for i in profiler._ext_calls})
        return iet, profiler

    def _specialize_iet(self, iet, **kwargs):
        """
        Transform the IET into a backend-specific representation, such as code
        to be executed on a GPU or through a lower-level system (e.g., YASK).
        """
        dle = kwargs.get("dle", configuration['dle'])

        # Apply the Devito Loop Engine (DLE) for loop optimization
        iet, state = transform(iet, *set_dle_mode(dle))

        self._func_table.update(OrderedDict([(i.name, MetaCall(i, True))
                                             for i in state.efuncs]))
        self._dimensions.extend(state.dimensions)
        self._includes.extend(state.includes)

        return iet

    def _finalize(self, iet, parameters):
        iet = iet_insert_decls(iet, parameters)
        iet = iet_insert_casts(iet, parameters)

        # Now do the same to each ElementalFunction
        for k, (root, local) in list(self._func_table.items()):
            if local:
                body = iet_insert_decls(root.body, root.parameters)
                body = iet_insert_casts(body, root.parameters)
                self._func_table[k] = MetaCall(root._rebuild(body=body), True)

        return iet

    # Arguments processing

    def _prepare_arguments(self, **kwargs):
        """
        Process runtime arguments passed to ``.apply()` and derive
        default values for any remaining arguments.
        """
        overrides, defaults = split(self.input, lambda p: p.name in kwargs)
        # Process data-carrier overrides
        args = ReducerMap()
        for p in overrides:
            args.update(p._arg_values(**kwargs))
            try:
                args = ReducerMap(args.reduce_all())
            except ValueError:
                raise ValueError("Override `%s` is incompatible with overrides `%s`" %
                                 (p, [i for i in overrides if i.name in args]))
        # Process data-carrier defaults
        for p in defaults:
            if p.name in args:
                # E.g., SubFunctions
                continue
            for k, v in p._arg_values(**kwargs).items():
                if k in args and args[k] != v:
                    raise ValueError("Default `%s` is incompatible with other args as "
                                     "`%s=%s`, while `%s=%s` is expected. Perhaps you "
                                     "forgot to override `%s`?" %
                                     (p, k, v, k, args[k], p))
                args[k] = v
        args = args.reduce_all()

        # All DiscreteFunctions should be defined on the same Grid
        grids = {getattr(p, 'grid', None) for p in overrides + defaults} - {None}
        if len(grids) > 1 and configuration['mpi']:
            raise ValueError("Multiple Grids found")
        try:
            grid = grids.pop()
        except KeyError:
            grid = None

        # Process Dimensions
        # A topological sorting is used so that derived Dimensions are processed after
        # their parents (note that a leaf Dimension can have an arbitrary long list of
        # ancestors)
        dag = DAG(self.dimensions,
                  [(i, i.parent) for i in self.dimensions if i.is_Derived])
        for d in reversed(dag.topological_sort()):
            args.update(d._arg_values(args, self._dspace[d], grid, **kwargs))

        # Process Objects (which may need some `args`)
        for o in self.objects:
            args.update(o._arg_values(args, **kwargs))

        # Sanity check
        for p in self.parameters:
            p._arg_check(args, self._dspace[p])
        for d in self.dimensions:
            if d.is_Derived:
                d._arg_check(args, self._dspace[p])

        # Turn arguments into a format suitable for the generated code
        # E.g., instead of NumPy arrays for Functions, the generated code expects
        # pointers to ctypes.Struct
        for p in self.parameters:
            try:
                args.update(kwargs.get(p.name, p)._arg_as_ctype(args, alias=p))
            except AttributeError:
                # User-provided floats/ndarray obviously do not have `_arg_as_ctype`
                args.update(p._arg_as_ctype(args, alias=p))

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

        # Attach `grid` to the arguments map
        args = ArgumentsMap(grid, **args)

        return args

    def _postprocess_arguments(self, args, **kwargs):
        """Process runtime arguments upon returning from ``.apply()``."""
        for p in self.parameters:
            try:
                p._arg_apply(args[p.name], args[p.coordinates.name], kwargs.get(p.name))
            except AttributeError:
                p._arg_apply(args[p.name], kwargs.get(p.name))

    @cached_property
    def _known_arguments(self):
        """The arguments that can be passed to ``apply`` when running the Operator."""
        ret = set.union(*[set(i._arg_names) for i in self.input + self.dimensions])
        return tuple(sorted(ret))

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

    # JIT compilation

    @cached_property
    def _soname(self):
        """A unique name for the shared object resulting from JIT compilation."""
        return Signer._digest(self, configuration)

    def _compile(self):
        """
        JIT-compile the C code generated by the Operator.

        It is ensured that JIT compilation will only be performed once per
        Operator, reagardless of how many times this method is invoked.
        """
        if self._lib is None:
            self._compiler.jit_compile(self._soname, str(self.ccode))

    @property
    def cfunction(self):
        """The JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:
            self._compile()
            self._lib = self._compiler.load(self._soname)
            self._lib.name = self._soname

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            # Associate a C type to each argument for runtime type check
            self._cfunction.argtypes = [i._C_ctype for i in self.parameters]

        return self._cfunction

    # Execution and profiling

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
        args = self.arguments(**kwargs)

        # Invoke kernel function with args
        arg_values = [args[p.name] for p in self.parameters]
        try:
            cfunction = self.cfunction
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
        return self._profile_output(args)

    def _profile_output(self, args):
        """Produce a performance summary of the profiled sections."""
        # Rounder to 2 decimal places
        fround = lambda i: ceil(i * 100) / 100

        info("Operator `%s` run in %.2f s" % (self.name,
                                              fround(self._profiler.py_timers['apply'])))

        summary = self._profiler.summary(args, self._dtype, reduce_over='apply')

        if summary.globals:
            indent = " "*2

            perf("Global performance indicators")

            # With MPI enabled, the 'vanilla' entry contains "cross-rank" performance data
            v = summary.globals.get('vanilla')
            if v is not None:
                gflopss = "%.2f GFlops/s" % fround(v.gflopss)
                gpointss = "%.2f GPts/s" % fround(v.gpointss) if v.gpointss else None
                metrics = ", ".join(i for i in [gflopss, gpointss] if i is not None)
                perf("%s* Operator `%s` with OI=%.2f computed in %.2f s [%s]" %
                     (indent, self.name, fround(v.oi), fround(v.time), metrics))

            v = summary.globals.get('fdlike')
            if v is not None:
                perf("%s* Achieved %.2f FD-GPts/s" % (indent, v.gpointss))

            perf("Local performance indicators")
        else:
            indent = ""

        # Emit local, i.e. "per-rank" performance. Without MPI, this is the only
        # thing that will be emitted
        for k, v in summary.items():
            rank = "[rank%d]" % k.rank if k.rank is not None else ""
            gflopss = "%.2f GFlops/s" % fround(v.gflopss)
            gpointss = "%.2f GPts/s" % fround(v.gpointss) if v.gpointss else None
            metrics = ", ".join(i for i in [gflopss, gpointss] if i is not None)
            itershapes = [",".join(str(i) for i in its) for its in v.itershapes]
            if len(itershapes) > 1:
                name = "%s%s<%s>" % (k.name, rank,
                                     ",".join("<%s>" % i for i in itershapes))
                perf("%s* %s with OI=%.2f computed in %.2f s [%s]" %
                     (indent, name, fround(v.oi), fround(v.time), metrics))
            elif len(itershapes) == 1:
                name = "%s%s<%s>" % (k.name, rank, itershapes[0])
                perf("%s* %s with OI=%.2f computed in %.2f s [%s]" %
                     (indent, name, fround(v.oi), fround(v.time), metrics))
            else:
                name = k.name
                perf("%s* %s%s computed in %.2f s"
                     % (indent, name, rank, fround(v.time)))

        perf("Configuration:  %s" % self._state['optimizations'])

        return summary

    # Misc properties

    @cached_property
    def _mem_summary(self):
        """
        The amount of data, in bytes, used by the Operator. This is provided as
        symbolic expressions, one symbolic expression for each memory scope (external,
        stack, heap).
        """
        roots = [self] + [i.root for i in self._func_table.values()]
        functions = [i for i in derive_parameters(roots) if i.is_Function]

        summary = {}

        external = [i.symbolic_shape for i in functions if i._mem_external]
        external = sum(reduce(mul, i, 1) for i in external)*self._dtype().itemsize
        summary['external'] = external

        heap = [i.symbolic_shape for i in functions if i._mem_heap]
        heap = sum(reduce(mul, i, 1) for i in heap)*self._dtype().itemsize
        summary['heap'] = heap

        stack = [i.symbolic_shape for i in functions if i._mem_stack]
        stack = sum(reduce(mul, i, 1) for i in stack)*self._dtype().itemsize
        summary['stack'] = stack

        summary['total'] = external + heap + stack

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
            self._compiler.save(self._soname, binary)
            self._lib = self._compiler.load(self._soname)
            self._lib.name = self._soname


# Misc helpers


class ArgumentsMap(dict):

    def __init__(self, grid, *args, **kwargs):
        super(ArgumentsMap, self).__init__(*args, **kwargs)
        self.grid = grid

    @property
    def comm(self):
        """The MPI communicator the arguments are collective over."""
        return self.grid.comm if self.grid is not None else MPI.COMM_NULL


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


def is_threaded(mode):
    return set_dle_mode(mode)[1].get('openmp', configuration['openmp'])

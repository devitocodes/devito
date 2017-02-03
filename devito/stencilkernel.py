from __future__ import absolute_import

import operator
from collections import OrderedDict, namedtuple
from ctypes import c_double, c_int
from functools import reduce
from hashlib import sha1
from os import path

import cgen as c
import numpy as np

from devito.compiler import (get_compiler_from_env, get_tmp_dir,
                             jit_compile_and_load)
from devito.dimension import Dimension
from devito.dle import transform
from devito.dse import indexify, rewrite
from devito.interfaces import SymbolicData
from devito.logger import error, info, warning
from devito.nodes import Block, Expression, Function, Iteration, TimedList
from devito.profiler import Profiler
from devito.tools import as_tuple
from devito.visitors import (EstimateCost, FindSections, FindSymbols,
                             IsPerfectIteration, MergeOuterIterations,
                             ResolveIterationVariable, SubstituteExpression,
                             Transformer, printAST)

__all__ = ['StencilKernel']


class StencilKernel(Function):

    """A special :class:`Function` to evaluate stencils through just-in-time
    compilation of C code.

    :param stencils: SymPy equation or list of equations that define the
                     stencil used to create the kernel of this Operator.
    :param kwargs: Accept the following entries: ::

        * name : Name of the kernel function - defaults to "Kernel".
        * subs : Dict or list of dicts containing the SymPy symbol
                 substitutions for each stencil respectively.
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
    def __init__(self, stencils, **kwargs):
        name = kwargs.get("name", "Kernel")
        subs = kwargs.get("subs", {})
        dse = kwargs.get("dse", "advanced")
        dle = kwargs.get("dle", "advanced")
        compiler = kwargs.get("compiler", None)

        # Default attributes required for compilation
        self.compiler = compiler or get_compiler_from_env()
        self.profiler = kwargs.get("profiler", Profiler(self.compiler.openmp))
        self._includes = ['stdlib.h', 'math.h', 'sys/time.h']
        self._lib = None
        self._cfunction = None

        # Convert stencil expressions into actual Nodes, going through the
        # Devito Symbolic Engine for flop optimization.
        stencils = stencils if isinstance(stencils, list) else [stencils]
        stencils = [indexify(s) for s in stencils]
        stencils = [s.xreplace(subs) for s in stencils]
        dse_state = rewrite(stencils, mode=dse)
        nodes = [Expression(s) for s in dse_state.exprs]

        # Wrap expressions with Iterations according to dimensions
        # TODO: This should probably be done more safely in a visitor
        # that tracks free and bound loop variables in the AST.
        for i, expr in enumerate(nodes):
            newexpr = expr
            offsets = newexpr.index_offsets
            for d in reversed(list(offsets.keys())):
                newexpr = Iteration(newexpr, dimension=d,
                                    limits=d.size, offsets=offsets[d])
            nodes[i] = newexpr

        # Merge Iterations iff outermost iterations agree
        nodes = MergeOuterIterations().visit(nodes)

        # Introduce profiling infrastructure
        mapper = {}
        self.sections = OrderedDict()
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
                        k = tuple(k.dim for k in itspace)
                        v = EstimateCost().visit(j)
                        self.sections[k] = Profile(lname, v.ops, v.mem)
                        break
        nodes = [Transformer(mapper).visit(Block(body=nodes))]

        # Now resolve and substitute dimensions for loop index variables
        subs = {}
        nodes = ResolveIterationVariable().visit(nodes, subs=subs)
        nodes = SubstituteExpression(subs=subs).visit(nodes)

        # Apply the Devito Loop Engine for loop optimization and finalize instantiation
        dle_state = transform(nodes, mode=set_dle_mode(dle, self.compiler),
                              compiler=self.compiler)
        body = dle_state.nodes
        parameters = FindSymbols().visit(nodes)
        parameters += [i.argument for i in dle_state.arguments]
        super(StencilKernel, self).__init__(name, body, 'int', parameters, ())

        # DLE might have introduced additional headers
        self._includes.extend(list(dle_state.includes))

        # Track the DSE and DLE output, as they may be useful later
        self._dse_state = dse_state
        self._dle_state = dle_state

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply defined stencil kernel to a set of data objects"""
        if len(args) <= 0:
            args = self.parameters

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
                    if dim.buffered:
                        # Check if provided as a keyword arg
                        size = kwargs.get(dim.name, None)
                        if size is None:
                            error("Unknown dimension size, please provide "
                                  "size via Kernel.apply(%s=<size>)" % dim.name)
                            raise RuntimeError('Dimension of unspecified size')
                        dim_sizes[dim] = size
                    elif dim in dim_sizes:
                        # Ensure size matches previously defined size
                        assert dim_sizes[dim] == data.shape[i]
                    else:
                        # Derive size from grid data shape and store
                        dim_sizes[dim] = data.shape[i]
                else:
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
            else:
                dle_arguments[i.argument] = dim_size
        dim_sizes.update(dle_arguments)
        # Insert loop size arguments from dimension values
        d_args = [d for d in arguments.values() if isinstance(d, Dimension)]
        for d in d_args:
            arguments[d.name] = dim_sizes[d]

        # Retrieve the data type of the arrays
        try:
            dtypes = [i.data.dtype for i in f_args]
            dtype = dtypes[0]
            if any(i != dtype for i in dtypes):
                warning("Found non-matching data types amongst the provided"
                        "symbolic arguments.")
            dtype_size = dtype.itemsize
        except IndexError:
            dtype_size = 1

        # Add profiler structs
        if self.profiler:
            cpointer = self.profiler.as_ctypes_pointer(Profiler.TIME)
            arguments[self.profiler.s_name] = cpointer

        # Invoke kernel function with args
        self.cfunction(*list(arguments.values()))

        # Summary of performance achieved
        info("="*79)
        for itspace, profile in self.sections.items():
            # Time
            elapsed = self.profiler.timings[profile.timer]
            # Flops
            niters = reduce(operator.mul, [i.size or dim_sizes[i] for i in itspace])
            flops = float(profile.ops*niters)
            gflops = flops/10**9
            # Compulsory traffic
            traffic = profile.memory*niters*dtype_size

            info("Section %s with OI=%.2f computed in %.2f s [Perf: %.2f GFlops/s]" %
                 (str(itspace), flops/traffic, elapsed, gflops/elapsed))
        info("="*79)

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
        elemental_functions = [e.ccode for e in self._dle_state.elemental_functions]
        elemental_functions += [blankline]

        # Generate file header with includes and definitions
        includes = [c.Include(i, system=False) for i in self._includes]
        includes += [blankline]
        profiling = [self.profiler.as_cgen_struct(Profiler.TIME), blankline]
        return c.Module(includes + profiling + elemental_functions + [kernel])

    @property
    def basename(self):
        """Generate the file basename path for auto-generated files

        The basename is generated from the hash string of the kernel,
        which is base on the final expressions and iteration symbols.

        :returns: The basename path as a string
        """
        expr_string = printAST(self.body, verbose=True)
        expr_string += printAST(self._dle_state.elemental_functions, verbose=True)
        hash_key = sha1(expr_string.encode()).hexdigest()

        return path.join(get_tmp_dir(), hash_key)

    @property
    def cfunction(self):
        """Returns the JIT-compiled C function as a ctypes.FuncPtr object

        Note that this invokes the JIT compilation toolchain with the
        compiler class derived in the constructor

        :returns: The generated C function
        """
        if self._lib is None:
            self._lib = jit_compile_and_load(self.ccode, self.basename,
                                             self.compiler)
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


"""
A dict of standard names to be used for code generation
"""
cnames = {
    'loc_timer': 'loc_timer',
    'glb_timer': 'glb_timer'
}

"""
A helper to track profiled sections of code.
"""
Profile = namedtuple('Profile', 'timer ops memory')


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

from __future__ import absolute_import

import operator
from collections import OrderedDict
from ctypes import c_double, c_int
from functools import reduce
from hashlib import sha1
from os import path

import cgen as c
import numpy as np

from devito.compiler import (get_compiler_from_env, get_tmp_dir,
                             jit_compile_and_load)
from devito.dimension import Dimension
from devito.dse.inspection import estimate_cost, estimate_memory, indexify
from devito.dse.symbolics import rewrite
from devito.interfaces import SymbolicData
from devito.logger import error, info
from devito.nodes import Block, Expression, Iteration, Timer
from devito.profiler import Profiler
from devito.visitors import (
    FindSections, IsPerfectIteration, Transformer, FindSymbols,
    SubstituteExpression, ResolveIterationVariable
)

__all__ = ['StencilKernel']


class StencilKernel(object):

    _includes = ['stdlib.h', 'math.h', 'sys/time.h',
                 'xmmintrin.h', 'pmmintrin.h']

    """Code generation class, alternative to Propagator

    :param stencils: SymPy equation or list of equations that define the
                     stencil used to create the kernel of this Operator.
    :param kwargs: Accept the following entries: ::

        * name : Name of the kernel function - defaults to "Kernel".
        * subs : Dict or list of dicts containing the SymPy symbol
                 substitutions for each stencil respectively.
        * dse : Use the Devito Symbolic Engine to optimize the expressions -
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
        compiler = kwargs.get("compiler", None)

        # Default attributes required for compilation
        self.name = name
        self.compiler = compiler or get_compiler_from_env()
        self.profiler = kwargs.get("profiler", Profiler(self.compiler.openmp))
        self._lib = None
        self._cfunction = None

        # Normalize stencils
        stencils = stencils if isinstance(stencils, list) else [stencils]
        stencils = [indexify(s) for s in stencils]
        stencils = [s.xreplace(subs) for s in stencils]
        stencils = rewrite(stencils, mode=dse).exprs
        self.expressions = [Expression(s) for s in stencils]

        # Wrap expressions with Iterations according to dimensions
        for i, expr in enumerate(self.expressions):
            newexpr = expr
            offsets = newexpr.index_offsets
            for d in reversed(expr.dimensions):
                newexpr = Iteration(newexpr, dimension=d,
                                    limits=d.size, offsets=offsets[d])
            self.expressions[i] = newexpr

        # TODO: Merge Iterations iff outermost variables agree

        # Introduce timers for profiling (only perfect nests are timed)
        mapper = {}
        for i, expr in enumerate(self.expressions):
            for itspace in FindSections().visit(expr).keys():
                for j in itspace:
                    if IsPerfectIteration().visit(j) and j not in mapper:
                        # Insert `Timer` block. This should come from
                        # the profiler, but we do this manually for now.
                        lname = 'loop_%s' % j.index
                        mapper[j] = Timer(gname=self.profiler.t_name,
                                          lname=lname, body=j)
                        self.profiler.t_fields += [(lname, c_double)]
                        break
        self.expressions = [Transformer(mapper).visit(Block(body=self.expressions))]

        # Now resolve and substitute dimensions for loop index variables
        subs = {}
        self.expressions = ResolveIterationVariable().visit(self.expressions, subs=subs)
        self.expressions = SubstituteExpression(subs=subs).visit(self.expressions)

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply defined stencil kernel to a set of data objects"""
        if len(args) <= 0:
            args = self.signature

        # Map of required arguments and actual dimension sizes
        arguments = OrderedDict([(arg.name, arg) for arg in self.signature])
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
        # Insert loop size arguments from dimension values
        d_args = [d for d in arguments.values() if isinstance(d, Dimension)]
        for d in d_args:
            arguments[d.name] = dim_sizes[d]

        # Add profiler structs
        if self.profiler:
            cpointer = self.profiler.as_ctypes_pointer(Profiler.TIME)
            arguments[self.profiler.s_name] = cpointer

        # Invoke kernel function with args
        self.cfunction(*list(arguments.values()))

        # Summary of performance achieved
        for itspace, expressions in self._sections.items():
            stencils = [e.stencil for e in expressions]
            niters = reduce(operator.mul, [i.size or dim_sizes[i] for i in itspace])
            flops = float(estimate_cost(stencils)*niters)
            gflops = flops/10**9
            # TODO: need to tweak the calculation below once padding is in
            traffic = estimate_memory(stencils)*niters
            info("Computed block %s in X s [OI=%.2f, Perf=%.3f GFlops/s]" %
                 (str(itspace), flops/traffic, gflops))

    @property
    def signature(self):
        """List of data objects that define the kernel signature

        :returns: List of unique data objects required by the kernel
        """
        return FindSymbols().visit(self.expressions)

    @property
    def ccode(self):
        """Returns the C code generated by this kernel.

        This function generates the internal code block from Iteration
        and Expression objects, and adds the necessary template code
        around it.
        """
        blankline = c.Line("")

        # Generate argument signature
        header_vars = [v.decl if isinstance(v, Dimension) else
                       c.Pointer(c.POD(v.dtype, '%s_vec' % v.name))
                       for v in self.signature]
        header_vars += [c.Pointer(c.Value('struct %s' % self.profiler.s_name,
                                          self.profiler.t_name))]
        header = c.FunctionDeclaration(c.Value('int', self.name), header_vars)

        # Generate data casts
        functions = [f for f in self.signature if isinstance(f, SymbolicData)]
        cast_shapes = [(f, ''.join(["[%s]" % i.ccode for i in f.indices[1:]]))
                       for f in functions]
        casts = [c.Initializer(c.POD(v.dtype, '(*%s)%s' % (v.name, shape)),
                               '(%s (*)%s) %s' % (c.dtype_to_ctype(v.dtype),
                                                  shape, '%s_vec' % v.name))
                 for v, shape in cast_shapes]

        # Gnerate function body with all the trimmings
        extra = [c.Comment('Force flushing of denormals to zero in hardware'),
                 c.Line('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);'),
                 c.Line('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);')]
        denormal = [c.Block(extra)]
        body = [e.ccode for e in self.expressions]
        ret = [c.Statement("return 0")]
        kernel = c.FunctionBody(header, c.Block(casts + denormal + body + ret))

        # Generate file header with includes and definitions
        includes = [c.Include(i, system=False) for i in self._includes]
        includes += [blankline]
        profiling = [self.profiler.as_cgen_struct(Profiler.TIME), blankline]
        return c.Module(includes + profiling + [kernel])

    @property
    def basename(self):
        """Generate the file basename path for auto-generated files

        The basename is generated from the hash string of the kernel,
        which is base on the final expressions and iteration symbols.

        :returns: The basename path as a string
        """
        expr_string = "\n".join([str(e) for e in self.expressions])
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
                for v in self.signature]

    @property
    def _sections(self):
        """Return the sections of the StencilKernel as a map from iteration
        spaces to expressions therein embedded. For example, given the loop tree:

            .. code-block::
               Iteration t
                 Iteration p
                   expr0
                 Iteration x
                   Iteration y
                     expr1
                     expr2
                 Iteration s
                   expr3

        Return the ordered map: ::

            {(t, p): [expr0], (t, x, y): [expr1, expr2], (t, s): [expr3]}
        """
        sections = FindSections().visit(self.expressions)
        return OrderedDict([(tuple(i.dim for i in k), v) for k, v in sections.items()])


"""
A dict of standard names to be used for code generation
"""
cnames = {
    'loc_timer': 'loc_timer',
    'glb_timer': 'glb_timer'
}

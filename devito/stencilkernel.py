from collections import OrderedDict
from ctypes import c_int
from hashlib import sha1
from itertools import chain
from os import path

import cgen as c
import numpy as np

from devito.compiler import (get_compiler_from_env, get_tmp_dir,
                             jit_compile_and_load)
from devito.expression import Expression
from devito.interfaces import SymbolicData
from devito.iteration import Iteration, IterationBound
from devito.logger import error
from devito.tools import filter_ordered

__all__ = ['StencilKernel']


class StencilKernel(object):
    """Code generation class, alternative to Propagator

    :param stencils: SymPy equation or list of equations that define the
                     stencil used to create the kernel of this Operator.
    :param name: Name of the kernel function - defaults to "Kernel"
    :param subs: Dict or list of dicts containing the SymPy symbol
                 substitutions for each stencil respectively.
    :param compiler: Compiler class used to perform JIT compilation.
                     If not provided, the compiler will be inferred from the
                     environment variable DEVITO_ARCH, or default to GNUCompiler.
    """

    def __init__(self, stencils, name="Kernel", subs=None, compiler=None):
        # Default attributes required for compilation
        self.name = name
        self.compiler = compiler or get_compiler_from_env()
        self._lib = None
        self._cfunction = None
        # Ensure we always deal with Expression lists
        stencils = stencils if isinstance(stencils, list) else [stencils]
        self.expressions = [Expression(s) for s in stencils]

        # Lower all expressions to "indexed" API
        for e in self.expressions:
            e.indexify()

        # Apply supplied substitutions
        if subs is not None:
            for expr in self.expressions:
                expr.substitute(subs)

        # Wrap expressions with Iterations according to dimensions
        for i, expr in enumerate(self.expressions):
            newexpr = expr
            offsets = newexpr.index_offsets
            for d in reversed(expr.dimensions):
                newexpr = Iteration(newexpr, dimension=d,
                                    limits=d.size, offsets=offsets[d])
            self.expressions[i] = newexpr

        # TODO: Merge Iterations iff outermost variables agree

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
        d_args = [d for d in arguments.values() if isinstance(d, IterationBound)]
        for d in d_args:
            arguments[d.name] = dim_sizes[d.dim]

        # Invoke kernel function with args
        self.cfunction(*list(arguments.values()))

    @property
    def signature(self):
        """List of data objects that define the kernel signature

        :returns: List of unique data objects required by the kernel
        """
        signature = [e.signature for e in self.expressions]
        return filter_ordered(chain(*signature))

    @property
    def ccode(self):
        """Returns the C code generated by this kernel.

        This function generates the internal code block from Iteration
        and Expression objects, and adds the necessary template code
        around it.
        """
        header_vars = [v.ccode if isinstance(v, IterationBound) else
                       c.Pointer(c.POD(v.dtype, '%s_vec' % v.name))
                       for v in self.signature]
        header = c.FunctionDeclaration(c.Value('int', self.name), header_vars)
        functions = [v for v in self.signature if not isinstance(v, IterationBound)]
        cast_shapes = [(f, ''.join(["[%s_size]" % i.name if i.size is None else i.size
                                    for i in f.indices[1:]]))
                       for f in functions]
        casts = [c.Initializer(c.POD(v.dtype, '(*%s)%s' % (v.name, shape)),
                               '(%s (*)%s) %s' % (c.dtype_to_ctype(v.dtype),
                                                  shape, '%s_vec' % v.name))
                 for v, shape in cast_shapes]
        body = [e.ccode for e in self.expressions]
        ret = [c.Statement("return 0")]
        return c.FunctionBody(header, c.Block(casts + body + ret))

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
        return [c_int if isinstance(v, IterationBound) else
                np.ctypeslib.ndpointer(dtype=v.dtype, flags='C')
                for v in self.signature]

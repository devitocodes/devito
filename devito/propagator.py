from __future__ import absolute_import

import operator
from collections import Iterable, defaultdict
from functools import reduce
from hashlib import sha1
from os import path
from random import randint

import numpy as np
from sympy import Indexed, IndexedBase, symbols
from sympy.utilities.iterables import postorder_traversal

import devito.cgen_wrapper as cgen
from devito.codeprinter import ccode
from devito.compiler import (IntelMICCompiler, get_compiler_from_env,
                             get_tmp_dir, jit_compile_and_load)
from devito.dimension import t, x, y, z
from devito.dse.inspection import retrieve_dtype
from devito.dse.symbolics import _temp_prefix
from devito.expression import Expression
from devito.function_manager import FunctionDescriptor, FunctionManager
from devito.iteration import Iteration
from devito.logger import info
from devito.profiler import Profiler
from devito.tools import flatten


class Propagator(object):

    """
    Propagator objects derive and encode C kernel code according
    to a given set of stencils, variables and time-stepping options.

    A kernel consists of a time loop wrapping three sections of code, called
    ``pre_stencils``, ``loop_body``, and ``post_stencils``.
    ``loop_body`` encodes the spatial loops and thus includes the stencils;
    the other two sections represent actions to be performed before and after
    the stencil computation, respectively.

    :param name: Name of the propagator kernel
    :param nt: Number of timesteps to execute
    :param shape: Shape of the data buffer over which to execute
    :param stencils: List of :class:`sympy.Eq` used to create the kernel
    :param spc_border: Number of spatial padding layers
    :param time_order: Order of the time discretisation
    :param time_dim: Symbol that defines the time dimension
    :param space_dims: List of symbols that define the space dimensions
    :param forward: Flag indicating whether to execute forward in time
    :param compiler: Compiler class used to perform JIT compilation.
                     If not provided, the compiler will be inferred from the
                     environment variable DEVITO_ARCH, or default to GNUCompiler
    :param profile: Flag to enable performance profiling
    :param cache_blocking: Block sizes used for cache clocking. Can be either a single
                           number used for all dimensions except inner most or a list
                           explicitly stating block sizes for each dimension
                           Set cache_blocking to None to skip blocking on that dim
                           Set cache_blocking to AutoTuner instance, to use auto tuned
                           tune block sizes
    """

    def __init__(self, name, nt, shape, stencils, spc_border=0, time_order=0,
                 time_dim=None, space_dims=None, dtype=np.float32, forward=True,
                 compiler=None, profile=False, cache_blocking=None):
        self.stencils = stencils
        self.dtype = dtype
        self.time_order = time_order
        self.spc_border = spc_border
        self.loop_body = None
        # Default time and space symbols if not provided
        self.time_dim = time_dim or t
        if not space_dims:
            space_dims = (x, z) if len(shape) == 2 else (x, y, z)[:len(shape)]
        self.space_dims = tuple(space_dims)
        self.shape = shape

        # Internal flags and meta-data
        self._forward = forward
        self.t_replace = {}
        self.time_steppers = []
        self.time_order = time_order
        self.nt = nt
        self.time_loop_stencils_b = []
        self.time_loop_stencils_a = []

        # Map objects in the mathematical model (e.g., x, t)
        # to C code (e.g., loop indices)
        symbol = lambda i: symbols("i%d" % i)
        self._mapper = {d: symbol(i) for i, d in enumerate(self.space_dims, 1)}
        self._mapper[self.time_dim] = symbol(len(self.space_dims)+1)

        # Start with the assumption that the propagator needs to save
        # the field in memory at every time step
        self._save = True

        # Which function parameters need special (non-save) time stepping and which don't
        self.save_vars = {}
        self.fd = FunctionDescriptor(name)
        self._time_step = 1 if forward else -1
        self._space_loop_limits = {}

        for i, dim in enumerate(self.space_dims):
            self._space_loop_limits[dim] = (spc_border, shape[i] - spc_border)

        # Derive JIT compilation infrastructure
        self.compiler = compiler or get_compiler_from_env()
        self.mic_flag = isinstance(self.compiler, IntelMICCompiler)
        self.sub_stencils = []

        # Settings for performance profiling
        self.profile = profile
        # Profiler needs to know whether openmp is set
        self.profiler = Profiler(self.compiler.openmp, self.dtype)

        # Cache blocking and block sizes
        self.cache_blocking = cache_blocking
        self.block_sizes = []
        self.shape = shape
        self.spc_border = spc_border

        # Cache C code, lib and function objects
        self._ccode = None
        self._lib = None
        self._cfunction = None

    def run(self, args, verbose=True):
        if self.profile:
            self.fd.add_struct_param(self.profiler.t_name, "profiler")

        f = self.cfunction

        # appends block sizes if cache blocking
        args += [block for block in self.block_sizes if block]

        if self.profile:
            args.append(self.profiler.as_ctypes_pointer(Profiler.TIME))

        if isinstance(self.compiler, IntelMICCompiler):
            # Off-load propagator kernel via pymic stream
            self.compiler._stream.invoke(f, *args)
            self.compiler._stream.sync()
        else:
            f(*args)

        if self.profile:
            if verbose:
                shape = str(self.shape).replace(', ', ' x ')
                cb = str(self.block_sizes) if self.cache_blocking else 'None'
                info("Shape: %s - Cache Blocking: %s" % (shape, cb))
                info("Time: %f s (%s MCells/s)" % (self.total_time, self.mcells))
            key = LOOP_BODY.name
            info("Stencil: %f OI, %.2f GFlops/s (time: %f s)" %
                 (self.oi[key], self.gflopss[key], self.timings[key]))

    @property
    def mcells(self):
        """
        Calculate how many MCells are computed, on average, in a second (the
        quantity is rounded to the closest unit).
        """
        itspace = map(lambda dim: dim - self.spc_border * 2, self.shape)
        return int(round(self.nt * np.prod(list(itspace))) / (self.total_time * 10**6))

    @property
    def basename(self):
        """Generate a unique basename path for auto-generated files

        The basename is generated by hashing grid variables (fd.params)
        and made unique by the addition of a random salt value

        :returns: The basename path as a string
        """
        string = "%s-%s" % (str(self.fd.params), randint(0, 100000000))

        return path.join(get_tmp_dir(), sha1(string.encode()).hexdigest())

    @property
    def ccode(self):
        """Returns the auto-generated C code as a string

        :returns: The code"""
        if self._ccode is None:
            manager = FunctionManager([self.fd], mic_flag=self.mic_flag,
                                      openmp=self.compiler.openmp)
            # For some reason we need this call to trigger fd.body
            self.get_fd()

            if self.profile:
                manager.add_struct_definition(self.profiler.as_cgen_struct(Profiler.TIME))

            self._ccode = str(manager.generate())

        return self._ccode

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
            self._cfunction = getattr(self._lib, self.fd.name)

            if not self.mic_flag:
                self._cfunction.argtypes = self.fd.argtypes

        return self._cfunction

    @property
    def timings(self):
        return self.profiler.timings

    @property
    def oi(self):
        """Summary of operational intensities, by code section."""

        gflops_per_section = self.gflops
        bytes_per_section = self.traffic()
        oi_per_section = {}

        for i, subsection in enumerate(self.time_loop_stencils_b):
            key = "%s%d" % (PRE_STENCILS.name, i)
            oi_per_section[key] = 1.0*gflops_per_section[key]/bytes_per_section[key]

        key = LOOP_BODY.name
        oi_per_section[key] = 1.0*gflops_per_section[key]/bytes_per_section[key]

        for i, subsection in enumerate(self.time_loop_stencils_a):
            key = "%s%d" % (POST_STENCILS.name, i)
            oi_per_section[key] = 1.0*gflops_per_section[key]/bytes_per_section[key]

        return oi_per_section

    @property
    def niters(self):
        """Summary of loop iterations, by code section."""

        niters_per_section = {}

        with_time_loop = lambda iters: self.nt*iters

        for i, subsection in enumerate(self.time_loop_stencils_b):
            key = "%s%d" % (PRE_STENCILS.name, i)
            niters = subsection.limits[1] if isinstance(subsection, Iteration) else 1
            niters_per_section[key] = with_time_loop(niters)

        key = LOOP_BODY.name
        niters = reduce(operator.mul,
                        [j - i for i, j in self._space_loop_limits.values()])
        niters_per_section[key] = with_time_loop(niters)

        for i, subsection in enumerate(self.time_loop_stencils_a):
            key = "%s%d" % (POST_STENCILS.name, i)
            niters = subsection.limits[1] if isinstance(subsection, Iteration) else 1
            niters_per_section[key] = with_time_loop(niters)

        return niters_per_section

    def traffic(self, mode='realistic'):
        """Summary of Bytes moved between CPU (last level cache) and DRAM,
        by code section.

        :param mode: Several estimates are possible: ::

            * ideal: also known as "compulsory traffic", which is the minimum
                number of bytes to be moved (ie, models an infinite cache)
            * ideal_with_stores: like ideal, but a data item which is both read
                and written is counted twice (load + store)
            * realistic: assume that all datasets, even those that do not depend
                on time, need to be re-loaded at each time iteration
        """

        assert mode in ['ideal', 'ideal_with_stores', 'realistic']

        def access(symbol):
            assert isinstance(symbol, Indexed)
            # Irregular accesses (eg A[B[i]]) are counted as compulsory traffic
            if any(i.atoms(Indexed) for i in symbol.indices):
                return symbol
            else:
                return symbol.base

        def count(self, expressions):
            if mode in ['ideal', 'ideal_with_stores']:
                filter = lambda s: self.time_dim in s.atoms()
            else:
                filter = lambda s: s
            reads = set(flatten([e.rhs.atoms(Indexed) for e in expressions]))
            writes = set(flatten([e.lhs.atoms(Indexed) for e in expressions]))
            reads = set([access(s) for s in reads if filter(s)])
            writes = set([access(s) for s in writes if filter(s)])
            if mode == 'ideal':
                return len(set(reads) | set(writes))
            else:
                return len(reads) + len(writes)

        niters = self.niters
        dsize = np.dtype(self.dtype).itemsize

        bytes_per_section = {}

        for i, subsection in enumerate(self.time_loop_stencils_b):
            key = "%s%d" % (PRE_STENCILS.name, i)
            if isinstance(subsection, Iteration):
                expressions = [e.stencil for e in subsection.expressions]
            else:
                expressions = subsection.stencil
            bytes_per_section[key] = dsize*count(self, expressions)*niters[key]

        key = LOOP_BODY.name
        bytes_per_section[key] = dsize*count(self, self.stencils)*niters[key]

        for i, subsection in enumerate(self.time_loop_stencils_a):
            key = "%s%d" % (POST_STENCILS.name, i)
            if isinstance(subsection, Iteration):
                expressions = [e.stencil for e in subsection.expressions]
            else:
                expressions = subsection.stencil
            bytes_per_section[key] = dsize*count(self, expressions)*niters[key]

        return bytes_per_section

    @property
    def gflops(self):
        """Summary of GFlops performed, by code section."""

        niters = self.niters

        gflops_per_iteration = self.profiler.gflops
        gflops_per_section = {}

        for i, subsection in enumerate(self.time_loop_stencils_b):
            key = "%s%d" % (PRE_STENCILS.name, i)
            gflops_per_section[key] = gflops_per_iteration[key]*niters[key]

        key = LOOP_BODY.name
        gflops_per_section[key] = gflops_per_iteration[key]*niters[key]

        for i, subsection in enumerate(self.time_loop_stencils_a):
            key = "%s%d" % (POST_STENCILS.name, i)
            gflops_per_section[key] = gflops_per_iteration[key]*niters[key]

        return gflops_per_section

    @property
    def gflopss(self):
        gflopss = {}
        for k, v in self.gflops.items():
            assert k in self.timings
            gflopss[k] = (float(v) / (10**9)) / self.timings[k]
        return gflopss

    @property
    def total_gflops(self):
        return sum(v / (10**9) for _, v in self.gflops.items())

    @property
    def total_time(self):
        return sum(v for _, v in self.timings.items())

    @property
    def total_gflopss(self):
        return self.total_gflops / self.total_time

    @property
    def save(self):
        """Indicates whether time history is saved.

        :returns: True if the time history is saved, else False.
        """
        return self._save

    @save.setter
    def save(self, save):
        """Function used to initialise the save parameter

        :param save: The new value of the save parameter
        """
        if save is not True:
            self.time_steppers = [symbols("t%d" % i) for i in range(self.time_order+1)]
            self.t_replace = {}

            for i, t_var in enumerate(reversed(self.time_steppers)):
                self.t_replace[self.time_dim - i*self._time_step] = t_var

            for i in range(1, len(self.time_steppers)):
                idx = self.time_dim + i*self._time_step
                self.t_replace[idx] = self.time_steppers[i - abs(self._time_step)]

        self._save = self._save and save

    @property
    def time_loop_limits(self):
        """Bounds for the time loop

        :returns: The bounds
        """
        if self._forward:
            loop_limits = (0, self.nt)
        else:
            loop_limits = (self.nt-1, -1)

        return loop_limits

    def sympy_to_cgen(self, stencils):
        """Converts sympy stencils to cgen statements

        :param stencils: A list of stencils to be converted
        :returns: :class:`cgen.Block` containing the converted kernel
        """
        declarations = []
        declared = defaultdict(bool)
        for eqn in stencils:
            s_lhs = str(eqn.lhs)
            if s_lhs.find(_temp_prefix) is not -1 and not declared[s_lhs]:
                expr_dtype = retrieve_dtype(eqn.rhs) or self.dtype
                declared[s_lhs] = True
                value = cgen.Value(cgen.dtype_to_ctype(expr_dtype), ccode(eqn.lhs))
                declarations.append(value)

        stmts = [self.convert_equality_to_cgen(x) for x in stencils]

        for idx, dec in enumerate(declarations):
            stmts[idx] = cgen.Assign(dec.inline(), stmts[idx].rvalue)

        return cgen.Block(stmts)

    def convert_equality_to_cgen(self, equality):
        """Convert given equality to :class:`cgen.Generable` statement

        :param equality: Given equality statement
        :returns: The resulting :class:`cgen.Generable` statement
        """
        if isinstance(equality, cgen.Generable):
            return equality
        elif isinstance(equality, Iteration):
            equality.substitute(self._mapper)
            return equality.ccode
        else:
            s_lhs = ccode(self.time_substitutions(equality.lhs).xreplace(self._mapper))
            s_rhs = self.time_substitutions(equality.rhs).xreplace(self._mapper)

            # appending substituted stencil,which is used to determine alignment pragma
            self.sub_stencils.append(s_rhs)

            s_rhs = ccode(s_rhs)
            if self.dtype is np.float32:
                s_rhs = str(s_rhs).replace("pow", "powf")
                s_rhs = str(s_rhs).replace("fabs", "fabsf")

            return cgen.Assign(s_lhs, s_rhs)

    def get_aligned_pragma(self, stencils, time_steppers):
        """
        Sets the alignment for the pragma.
        :param stencils: List of stencils.
        :param time_steppers: list of time stepper symbols
        """
        array_names = set()
        for item in flatten([stencil.free_symbols for stencil in stencils]):
            if (
                item not in self._mapper.values() + time_steppers
                and str(item).find(_temp_prefix) == -1
            ):
                array_names.add(item)
        if len(array_names) == 0:
            return cgen.Line("")
        else:
            return cgen.Pragma("%s(%s:64)" % (self.compiler.pragma_aligned,
                                              ", ".join([str(i) for i in array_names])
                                              ))

    def generate_loops(self):
        """Assuming that the variable order defined in init (#var_order) is the
        order the corresponding dimensions are layout in memory, the last variable
        in that definition should be the fastest varying dimension in the arrays.
        Therefore reverse the list of dimensions, making the last variable in
        #var_order (z in the 3D case) vary in the inner-most loop

        :param loop_body: Statement representing the loop body
        :returns: :class:`cgen.Block` representing the loop
        """

        if self.loop_body:
            time_invariants = []
            loop_body = self.loop_body
        elif self.time_order:
            time_invariants = [i for i in self.stencils if isinstance(i.lhs, Indexed)
                               and i.lhs.indices == self.space_dims]
            loop_body = [i for i in self.stencils if i not in time_invariants]
            loop_body = self.sympy_to_cgen(loop_body)
        else:
            time_invariants = []
            loop_body = self.sympy_to_cgen(self.stencils)

        # Space loops
        if not isinstance(loop_body, cgen.Block) or len(loop_body.contents) > 0:
            if self.cache_blocking is not None:
                self._decide_block_sizes()

                loop_body = self.generate_space_loops_blocking(loop_body)
            else:
                loop_body = self.generate_space_loops(loop_body)
        else:
            loop_body = []
        omp_master = [cgen.Pragma("omp master")] if self.compiler.openmp else []
        omp_single = [cgen.Pragma("omp single")] if self.compiler.openmp else []
        omp_parallel = [cgen.Pragma("omp parallel")] if self.compiler.openmp else []
        omp_for = [cgen.Pragma("omp for schedule(static)"
                               )] if self.compiler.openmp else []
        t_loop_limits = self.time_loop_limits
        t_var = str(self._mapper[self.time_dim])
        cond_op = "<" if self._forward else ">"

        if self.save is not True:
            # To cycle between array elements when we are not saving time history
            time_stepping = self.get_time_stepping()
        else:
            time_stepping = []
        if len(loop_body) > 0:
            loop_body = [cgen.Block(omp_for + loop_body)]

        # Generate code to be inserted outside of the space loops
        if time_invariants:
            ctype = cgen.dtype_to_ctype(self.dtype)
            getname = lambda i: i.lhs.base if isinstance(i.lhs, Indexed) else i.lhs
            values = {
                "type": ctype,
                "name": "%(name)s",
                "dsize": "".join("[%d]" % j for j in self.shape[:-1]),
                "size": "".join("[%d]" % j for j in self.shape)
            }
            declaration = "%(type)s (*%(name)s)%(dsize)s;" % values
            header = [cgen.Line(declaration % {'name': getname(i)})
                      for i in time_invariants]
            funcall = "posix_memalign((void**)&%(name)s, 64, sizeof(%(type)s%(size)s));"
            funcall = funcall % values
            funcalls = [cgen.Line(funcall % {'name': getname(i)})
                        for i in time_invariants]
            time_invariants = [self.convert_equality_to_cgen(i)
                               for i in time_invariants]
            time_invariants = self.generate_space_loops(cgen.Block(time_invariants),
                                                        full=True)
            time_invariants = [cgen.Block(funcalls + time_invariants)]
        else:
            header = []

        # Avoid denormal numbers
        extra = [cgen.Line('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);'),
                 cgen.Line('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);')]

        # Statements to be inserted into the time loop before the spatial loop
        pre_stencils = [self.time_substitutions(x)
                        for x in self.time_loop_stencils_b]
        pre_stencils = [self.convert_equality_to_cgen(x)
                        for x in self.time_loop_stencils_b]

        # Statements to be inserted into the time loop after the spatial loop
        post_stencils = [self.time_substitutions(x)
                         for x in self.time_loop_stencils_a]
        post_stencils = [self.convert_equality_to_cgen(x)
                         for x in self.time_loop_stencils_a]

        if self.profile:
            pre_stencils = list(flatten([self.profiler.add_profiling([s], "%s%d" %
                                         (PRE_STENCILS.name, i)) for i, s in
                                         enumerate(pre_stencils)]))
            post_stencils = list(flatten([self.profiler.add_profiling([s], "%s%d" %
                                          (POST_STENCILS.name, i)) for i, s in
                                          enumerate(post_stencils)]))

        initial_block = time_stepping + pre_stencils

        if initial_block:
            initial_block = omp_single + [cgen.Block(initial_block)]

        end_block = post_stencils

        if end_block:
            end_block = omp_single + [cgen.Block(end_block)]

        if self.profile:
            loop_body = self.profiler.add_profiling(loop_body, LOOP_BODY.name,
                                                    omp_flag=omp_master)

        loop_body = cgen.Block(initial_block + loop_body + end_block)

        loop_body = cgen.For(
            cgen.InlineInitializer(cgen.Value("int", t_var), str(t_loop_limits[0])),
            t_var + cond_op + str(t_loop_limits[1]),
            t_var + "+=" + str(self._time_step),
            loop_body
        )

        # Code to declare the time stepping variables (outside the time loop)
        def_time_step = [cgen.Value("int", t_var_def.name)
                         for t_var_def in self.time_steppers]
        body = cgen.Block(extra + time_invariants + def_time_step +
                          omp_parallel + [loop_body])

        return header + [body]

    def generate_space_loops(self, loop_body, full=False):
        """Generate list<cgen.For> for a non cache blocking space loop
        :param loop_body: Statement representing the loop body
        :returns: :list<cgen.For> a list of for loops
        """
        inner_most_dim = True

        for spc_var in reversed(list(self.space_dims)):
            dim_var = self._mapper[spc_var]
            loop_limits = self._space_loop_limits[spc_var]
            if full:
                loop_limits = (loop_limits[0] - self.spc_border,
                               loop_limits[1] + self.spc_border)
            loop_body = cgen.For(
                cgen.InlineInitializer(cgen.Value("int", dim_var), str(loop_limits[0])),
                str(dim_var) + "<" + str(loop_limits[1]),
                str(dim_var) + "++",
                loop_body
            )

            loop_body = self.add_inner_most_dim_pragma(inner_most_dim, self.space_dims,
                                                       loop_body)
            inner_most_dim = False
        return [loop_body]  # returns body as a list

    def generate_space_loops_blocking(self, loop_body):
        """Generate list<cgen.For> for a cache blocking space loop
        :param loop_body: Statement representing the loop body
        :returns: :list<cgen.For> a list of for loops
        """

        inner_most_dim = True
        orig_loop_body = loop_body

        omp_for = [cgen.Pragma("omp for schedule(static)"
                               )] if self.compiler.openmp else []

        for spc_var, block_size in reversed(list(zip(list(self.space_dims),
                                                     self.block_sizes))):
            orig_var = str(self._mapper[spc_var])
            block_var = orig_var + "b"
            loop_limits = self._space_loop_limits[spc_var]

            if block_size is not None:
                upper_limit_str = "%s+%sblock" % (block_var, orig_var)
                lower_limit_str = block_var
            else:
                lower_limit_str = str(loop_limits[0])
                upper_limit_str = str(loop_limits[1])

            loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", orig_var),
                                                        lower_limit_str),
                                 orig_var + "<" + upper_limit_str,
                                 orig_var + "++", loop_body)

            loop_body = self.add_inner_most_dim_pragma(inner_most_dim, self.space_dims,
                                                       loop_body)
            inner_most_dim = False

        remainder_counter = 0  # indicates how many remainder loops we need
        for spc_var, block_size in reversed(list(zip(list(self.space_dims),
                                                     self.block_sizes))):
            # if block size set to None do not block this dimension
            if block_size is not None:
                orig_var = str(self._mapper[spc_var])
                block_var = orig_var + "b"
                loop_limits = self._space_loop_limits[spc_var]

                block_size_str = orig_var + "block"
                upper_limit_str = "%d - (%d %% %s)" % (loop_limits[1],
                                                       loop_limits[1] - loop_limits[0],
                                                       block_size_str)

                loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", block_var),
                                                            str(loop_limits[0])),
                                     str(block_var) + "<" + upper_limit_str,
                                     str(block_var) + "+=" + block_size_str, loop_body)
                remainder_counter += 1

        full_remainder = []
        # weights for deciding remainder loop limit
        weights = self._decide_weights(self.block_sizes, remainder_counter)
        for i in range(remainder_counter):
            remainder_loop = orig_loop_body
            inner_most_dim = True

            for spc_var, block_size in reversed(list(zip(list(self.space_dims),
                                                         self.block_sizes))):
                orig_var = str(self._mapper[spc_var])
                loop_limits = self._space_loop_limits[spc_var]  # Full loop limits
                lower_limit_str = str(loop_limits[0])
                upper_limit_str = str(loop_limits[1])

                if block_size is not None:
                    if weights[orig_var] < 0:
                        # already blocked loop limits
                        upper_limit_str = "%d - (%d %% %s)" % (loop_limits[1],
                                                               loop_limits[1] -
                                                               loop_limits[0],
                                                               orig_var + "block")
                    elif weights[orig_var] == 0:
                        # remainder loop limits
                        lower_limit_str = "%d - (%d %% %s)" % (loop_limits[1],
                                                               loop_limits[1] -
                                                               loop_limits[0],
                                                               orig_var + "block")
                    weights[orig_var] += 1

                remainder_loop = cgen.For(cgen.InlineInitializer(cgen.Value("int",
                                                                            orig_var),
                                                                 lower_limit_str),
                                          str(orig_var) + "<" + upper_limit_str,
                                          str(orig_var) + "++", remainder_loop)

                remainder_loop = self.add_inner_most_dim_pragma(inner_most_dim,
                                                                self.space_dims,
                                                                remainder_loop)
                inner_most_dim = False

            full_remainder += omp_for
            full_remainder.append(remainder_loop)

        return [loop_body] + full_remainder if full_remainder else [loop_body]

    def _decide_block_sizes(self):
        """Decides block size checks whether args have been provided correctly

        :raises ValueError: If cache blocking parameters where passed incorrectly
        """
        if isinstance(self.cache_blocking, Iterable):
            if len(self.cache_blocking) == len(self.shape):
                self.block_sizes = self.cache_blocking
            else:
                raise ValueError("Cache blocking/block sizes have to be an array of the "
                                 "same size as the spacial domain or single int instance")
        else:
            # A single block size has been passed. Broadcast it to a list
            # We do not want to cache block outer most dim if one int value was passed
            self.block_sizes = [int(self.cache_blocking)] * (len(self.shape) - 1)
            self.block_sizes.append(None)

        for i, j in zip(self.block_sizes, self.space_dims):
            if i is not None:
                self.fd.add_value_param("%sblock" % self._mapper[j], np.int64)

    def add_inner_most_dim_pragma(self, inner_most_dim, space_dims, loop_body):
        """
        Adds pragma to inner most dim
        :param inner_most_dim: flag indicating whether its inner most dim
        :param space_dims: space dimensions of kernel
        :param loop_body: original loop body
        :return: cgen.Block - loop body with pragma
        """
        if inner_most_dim and len(space_dims) > 1:
            if self.compiler.openmp:
                pragma = [self.get_aligned_pragma(self.sub_stencils, self.time_steppers)]
            else:
                pragma = self.compiler.pragma_ivdep + self.compiler.pragma_nontemporal
            loop_body = cgen.Block(pragma + [loop_body])
        return loop_body

    def _decide_weights(self, block_sizes, remainder_counter):
        """
        Decide weights used for remainder loop limit calculations

        :param block_sizes: list of block sizes
        :param remainder_counter: int stating how many remainder loops are needed
        :return: dict of weights
        """
        key = lambda i: str(self._mapper[i])
        weights = {key(i): 0 for i in self.space_dims}

        if len(block_sizes) == 3 and remainder_counter > 1:
            if block_sizes[0] and block_sizes[1] and block_sizes[2]:
                weights[key(x)] = -1
                if remainder_counter == 3:
                    weights[key(y)] = -2
            elif (block_sizes[0] and block_sizes[1] and not block_sizes[2]) or\
                 (not block_sizes[0] and block_sizes[1] and block_sizes[2]):
                weights[key(y)] = -1
            else:
                weights[key(x)] = -1

        elif (len(block_sizes) == 2 and remainder_counter > 1 and
              block_sizes[0] and block_sizes[1]):
            weights[key(x)] = -1

        return weights

    def add_devito_param(self, param):
        """Setup relevant devito parameters

        :param param: Contains all devito parameters
        """
        save = True

        if hasattr(param, "save"):
            save = param.save

        self.add_param(param.name, param.shape, param.dtype, save)

    def add_param(self, name, shape, dtype, save=True):
        """Add a matrix parameter to the propagator

        :param name: Name of parameter
        :param shape: Shape of parameter
        :param dtype: Base type of parameter
        :param save: Indicates whether time history is saved
        :returns: :class:`sympy.IndexedBase` containing the matrix parameter
        """
        self.fd.add_matrix_param(name, shape, dtype)
        self.save = save
        self.save_vars[name] = save

        return IndexedBase(name, shape=shape)

    def add_scalar_param(self, name, dtype):
        """Add a scalar parameter to the propagator. E.g. int

        :param name: Name of parameter
        :param dtype: Type of parameter
        """
        self.fd.add_value_param(name, dtype)

        return symbols(name)

    def add_local_var(self, name, dtype):
        """Add a local scalar parameter

        :param name: Name of parameter
        :param dtype: Type of parameter
        :returns: The symbol associated with the parameter
        """
        self.fd.add_local_variable(name, dtype)

        return symbols(name)

    def get_fd(self):
        """Get a FunctionDescriptor that describes the code represented by this Propagator
        in the format that FunctionManager and JitManager can deal with it.
        Before calling, make sure you have either called set_jit_params
        or set_jit_simple already

        :returns: The resulting :class:`devito.function_manager.FunctionDescriptor`
        """

        self.fd.set_body(self.generate_loops())
        return self.fd

    def get_time_stepping(self):
        """Add the time stepping code to the loop

        :returns: A list of :class:`cgen.Statement` containing the time stepping code
        """
        ti = self._mapper[self.time_dim]
        body = []
        time_stepper_indices = range(self.time_order+1)
        first_time_index = 0
        step_backwards = -1

        if self._forward is not True:
            time_stepper_indices = reversed(time_stepper_indices)
            first_time_index = self.time_order
            step_backwards = 1

        for i in time_stepper_indices:
            lhs = self.time_steppers[i].name

            if i == first_time_index:
                rhs = ccode(ti % (self.time_order+1))
            else:
                rhs = ccode((self.time_steppers[i+step_backwards]+1) %
                            (self.time_order+1))

            body.append(cgen.Assign(lhs, rhs))

        return body

    def time_substitutions(self, sympy_expr):
        """This method checks through the sympy_expr to replace the time index with
        a cyclic index but only for variables which are not being saved in the time domain
        :param sympy_expr: The Sympy expression to process
        :returns: The expression after the substitutions
        """
        subs_dict = {}

        # For Iteration objects we apply time subs to the stencil list
        if isinstance(sympy_expr, Iteration):
            sympy_expr.expressions = [Expression(self.time_substitutions(s.stencil))
                                      for s in sympy_expr.expressions]
            return sympy_expr

        for arg in postorder_traversal(sympy_expr):
            if isinstance(arg, Indexed):
                is_saved = self.save_vars.get(str(arg.base.label), True)
                if not is_saved:
                    subs_dict[arg] = arg.xreplace(self.t_replace)

        return sympy_expr.xreplace(subs_dict)


class Section(object):

    """A code section in a stencil kernel."""

    def __init__(self, name):
        self.name = name


PRE_STENCILS = Section('pre_stencils')
LOOP_BODY = Section('loop_body')
POST_STENCILS = Section('post_stencils')

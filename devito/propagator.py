from collections import Iterable, defaultdict
from hashlib import sha1
from os import path
from random import randint

import numpy as np
from sympy import Indexed, IndexedBase, preorder_traversal, symbols
from sympy.utilities.iterables import postorder_traversal

import cgen_wrapper as cgen
from codeprinter import ccode
from devito.compiler import (IntelMICCompiler, get_compiler_from_env,
                             get_tmp_dir, jit_compile_and_load)
from devito.dimension import t, x, y, z
from devito.expression import Expression
from devito.function_manager import FunctionDescriptor, FunctionManager
from devito.iteration import Iteration
from devito.logger import logger
from devito.profiler import Profiler
from tools import flatten, get_optimal_block_size


class Propagator(object):
    """Propagator objects derive and encode C kernel code according
    to a given set of stencils, variables and time-stepping options

    :param name: Name of the propagator kernel
    :param nt: Number of timesteps to execute
    :param shape: Shape of the data buffer over which to execute
    :param stencils: List of :class:`sympy.Eq` used to create the kernel
    :param factorized: A map given by {string_name:sympy_object} for including factorized
                       terms
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
                           Set cache_blocking to 0 to use best guess based on architecture
                           Set cache_blocking to AutoTuner instance, to use auto tuned
                           tune block sizes
    """
    def __init__(self, name, nt, shape, stencils, factorized=None, spc_border=0,
                 time_order=0, time_dim=None, space_dims=None, dtype=np.float32,
                 forward=True, compiler=None, profile=False,
                 cache_blocking=None):
        self.stencils = stencils
        self.dtype = dtype
        self.factorized = factorized or {}
        self.time_order = time_order
        self.spc_border = spc_border

        # Default time and space symbols if not provided
        self.time_dim = time_dim or t
        self.space_dims = \
            space_dims or (x, z) if len(shape) == 2 else (x, y, z)[:len(shape)]
        self.shape = shape

        # Internal flags and meta-data
        self.loop_counters = list(symbols("i1 i2 i3 i4"))
        self._pre_kernel_steps = []
        self._post_kernel_steps = []
        self._forward = forward
        self.prep_variable_map()
        self.t_replace = {}
        self.time_steppers = []
        self.time_order = time_order
        self.nt = nt
        self.time_loop_stencils_b = []
        self.time_loop_stencils_a = []

        # Start with the assumption that the propagator needs to save
        # the field in memory at every time step
        self._save = True

        # Which function parameters need special (non-save) time stepping and which don't
        self.save_vars = {}
        self.fd = FunctionDescriptor(name)
        self.pre_loop = []
        self.post_loop = []
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
        self.profiler = Profiler(self.compiler.openmp)

        # suffix of temp variable used for loop reduction of corrosponding struct
        # member in Profiler struct when openmp is on
        self.reduction_list = ["kernel", "loop_body"] if self.compiler.openmp else []
        self.reduction_clause = (self.profiler
                                 .get_loop_reduction("+", self.reduction_list)
                                 if self.profile else "")

        # Cache blocking and block sizes
        self.cache_blocking = cache_blocking
        self.block_sizes = []
        self.shape = shape
        self.spc_border = spc_border

        # Cache C code, lib and function objects
        self._ccode = None
        self._lib = None
        self._cfunction = None

    def run(self, args):
        if self.profile:
            self.fd.add_struct_param(self.profiler.t_name, "profiler")
            self.fd.add_struct_param(self.profiler.f_name, "flops")

        f = self.cfunction

        # appends block sizes if cache blocking
        args += [block for block in self.block_sizes if block]

        if self.profile:
            args.append(self.profiler.as_ctypes_pointer(Profiler.TIME))
            args.append(self.profiler.as_ctypes_pointer(Profiler.FLOP))

        if isinstance(self.compiler, IntelMICCompiler):
            # Off-load propagator kernel via pymic stream
            self.compiler._stream.invoke(f, *args)
            self.compiler._stream.sync()
        else:
            f(*args)

        if self.profile:
            self.log_performance()

    @property
    def mcells(self):
        """Calculates MCELLS

        :returns: Mcells as int
        """
        iteration_space = map(lambda dim: dim - self.spc_border * 2, self.shape)
        return int(round(self.nt * np.prod(iteration_space)) /
                   (self.timings['kernel'] * 10 ** 6))

    @property
    def basename(self):
        """Generate a unique basename path for auto-generated files

        The basename is generated by hashing grid variables (fd.params)
        and made unique by the addition of a random salt value

        :returns: The basename path as a string
        """
        string = "%s-%s" % (str(self.fd.params), randint(0, 100000000))

        return path.join(get_tmp_dir(), sha1(string).hexdigest())

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
                manager.add_struct_definition(self.profiler.as_cgen_struct(Profiler.FLOP))

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
        return self.profiler.oi

    @property
    def oi_low(self):
        return self.profiler.oi_low

    @property
    def total_loads(self):
        return self.profiler.total_load_count

    @property
    def gflops(self):
        return self.profiler.gflops

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

    def log_performance(self):
        """Logs performance metrics"""
        shape_str = str(self.shape).replace(', ', ' x ')
        cb_str = ", blocks - %s " % str(self.block_sizes) \
            if self.cache_blocking else ' '

        logger.info("shape - %s%s:: %f sec - %s MCells/s - %.2f GFLOPS" %
                    (shape_str, cb_str, self.timings['kernel'],
                     self.mcells, self.gflops['kernel']))

    def get_number_of_loads(self):
        """Gets total number of loads which is used for optimal block size estimation
           Temp profiler needed as we want this info even when profiling is off

        :returns: Total number of loads
        """
        name = 'load_count'
        profiler = Profiler()
        at_stencils = [self.convert_equality_to_cgen(stencil)
                       for stencil in self.stencils]
        profiler.add_profiling(at_stencils, name)
        return profiler.total_load_count[name]

    def prep_variable_map(self):
        """Mapping from model variables (x, y, z, t) to loop variables (i1, i2, i3, i4)

        For now, i1 i2 i3 are assigned in the order
        the variables were defined in init( #var_order)
        """
        var_map = {}
        i = 0

        for dim in self.space_dims:
            var_map[dim] = symbols("i%d" % (i + 1))
            i += 1

        var_map[self.time_dim] = symbols("i%d" % (i + 1))
        self._var_map = var_map

    def sympy_to_cgen(self, stencils):
        """Converts sympy stencils to cgen statements

        :param stencils: A list of stencils to be converted
        :returns: :class:`cgen.Block` containing the converted kernel
        """

        factors = []
        if len(self.factorized) > 0:
            for name, term in zip(self.factorized.keys(), self.factorized):
                expr = self.factorized[name]
                self.add_local_var(name, self.dtype)
                sub = str(ccode(self.time_substitutions(expr).xreplace(self._var_map)))
                if self.dtype is np.float32:
                    factors.append(cgen.Assign(name, (sub.replace("pow", "powf")
                                                      .replace("fabs", "fabsf"))))
                else:
                    factors.append(cgen.Assign(name, sub))

        decl = []

        declared = defaultdict(bool)
        for eqn in stencils:
            s_lhs = str(eqn.lhs)
            if s_lhs.find("temp") is not -1 and not declared[s_lhs]:
                declared[s_lhs] = True
                decl.append(
                    cgen.Value(
                        cgen.dtype_to_ctype(self.expr_dtype(eqn.rhs)), ccode(eqn.lhs)
                    )
                )

        stmts = []

        for equality in stencils:
            stencil = self.convert_equality_to_cgen(equality)
            stmts.append(stencil)

        kernel = self._pre_kernel_steps
        kernel += stmts
        kernel += self._post_kernel_steps

        return cgen.Block(factors+decl+kernel)

    def expr_dtype(self, expr):
        """Gets the resulting dtype of an expression.

        :param expr: The expression

        :returns: The dtype. Defaults to `self.dtype` if none found.
        """
        dtypes = []

        for e in preorder_traversal(expr):
            if hasattr(e, 'dtype'):
                dtypes.append(e.dtype)

        if dtypes:
            return np.find_common_type(dtypes, [])
        else:
            return self.dtype

    def convert_equality_to_cgen(self, equality):
        """Convert given equality to :class:`cgen.Generable` statement

        :param equality: Given equality statement
        :returns: The resulting :class:`cgen.Generable` statement
        """
        if isinstance(equality, cgen.Generable):
            return equality
        elif isinstance(equality, Iteration):
            equality.substitute(self._var_map)
            return equality.ccode
        else:
            s_lhs = ccode(self.time_substitutions(equality.lhs).xreplace(self._var_map))
            s_rhs = self.time_substitutions(equality.rhs).xreplace(self._var_map)

            # appending substituted stencil,which is used to determine alignment pragma
            self.sub_stencils.append(s_rhs)

            s_rhs = ccode(s_rhs)
            if self.dtype is np.float32:
                s_rhs = str(s_rhs).replace("pow", "powf")
                s_rhs = str(s_rhs).replace("fabs", "fabsf")

            return cgen.Assign(s_lhs, s_rhs)

    def get_aligned_pragma(self, stencils, factorized, loop_counters, time_steppers,
                           reduction=None):
        """
        Sets the alignment for the pragma.
        :param stencils: List of stencils.
        :param factorized:  dict of factorized elements
        :param loop_counters: list of loop counter symbols
        :param time_steppers: list of time stepper symbols
        """
        array_names = set()
        for item in flatten([stencil.free_symbols for stencil in stencils]):
            if (
                str(item) not in factorized
                and item not in loop_counters + time_steppers
                and str(item).find("temp") == -1
            ):
                array_names.add(item)

        return cgen.Pragma("%s(%s:64)%s" % (self.compiler.pragma_aligned,
                                            ", ".join([str(i) for i in array_names]),
                                            self.reduction_clause))

    def generate_loops(self, loop_body):
        """Assuming that the variable order defined in init (#var_order) is the
        order the corresponding dimensions are layout in memory, the last variable
        in that definition should be the fastest varying dimension in the arrays.
        Therefore reverse the list of dimensions, making the last variable in
        #var_order (z in the 3D case) vary in the inner-most loop

        :param loop_body: Statement representing the loop body
        :returns: :class:`cgen.Block` representing the loop
        """

        # Space loops
        if self.cache_blocking is not None:
            self._decide_block_sizes()

            loop_body = self.generate_space_loops_blocking(loop_body)
        else:
            loop_body = self.generate_space_loops(loop_body)

        omp_master = [cgen.Pragma("omp master")] if self.compiler.openmp else []
        omp_single = [cgen.Pragma("omp single")] if self.compiler.openmp else []
        omp_parallel = [cgen.Pragma("omp parallel")] if self.compiler.openmp else []
        omp_for = [cgen.Pragma("omp for schedule(static)%s" %
                               self.reduction_clause)] if self.compiler.openmp else []
        t_loop_limits = self.time_loop_limits
        t_var = str(self._var_map[self.time_dim])
        cond_op = "<" if self._forward else ">"

        if self.save is not True:
            # To cycle between array elements when we are not saving time history
            time_stepping = self.get_time_stepping()
        else:
            time_stepping = []

        loop_body = [cgen.Block(omp_for + loop_body)]
        # Statements to be inserted into the time loop before the spatial loop
        time_loop_stencils_b = [self.time_substitutions(x)
                                for x in self.time_loop_stencils_b]
        time_loop_stencils_b = [self.convert_equality_to_cgen(x)
                                for x in self.time_loop_stencils_b]

        # Statements to be inserted into the time loop after the spatial loop
        time_loop_stencils_a = [self.time_substitutions(x)
                                for x in self.time_loop_stencils_a]
        time_loop_stencils_a = [self.convert_equality_to_cgen(x)
                                for x in self.time_loop_stencils_a]

        if self.profile:
            time_loop_stencils_a = self.profiler.add_profiling(time_loop_stencils_a,
                                                               "loop_stencils_a")
            time_loop_stencils_b = self.profiler.add_profiling(time_loop_stencils_b,
                                                               "loop_stencils_b")

        initial_block = time_stepping + time_loop_stencils_b

        if initial_block:
            initial_block = omp_single + [cgen.Block(initial_block)]

        end_block = time_loop_stencils_a

        if end_block:
            end_block = omp_single + [cgen.Block(end_block)]

        if self.profile:
            loop_body = self.profiler.add_profiling(loop_body, "loop_body",
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
        if self.profile:
            body = def_time_step + self.pre_loop +\
                [cgen.Statement(self.profiler
                                .get_loop_temp_var_decl("0", self.reduction_list))]\
                + omp_parallel + [loop_body] +\
                [cgen.Statement(s) for s in
                 self.profiler.get_loop_flop_update(self.reduction_list)]\
                + self.post_loop
        else:
            body = def_time_step + self.pre_loop\
                + omp_parallel + [loop_body] + self.post_loop

        if self.profile:
            body = self.profiler.add_profiling(body, "kernel")

        return cgen.Block(body)

    def generate_space_loops(self, loop_body):
        """Generate list<cgen.For> for a non cache blocking space loop
        :param loop_body: Statement representing the loop body
        :returns: :list<cgen.For> a list of for loops
        """
        inner_most_dim = True

        for spc_var in reversed(list(self.space_dims)):
            dim_var = self._var_map[spc_var]
            loop_limits = self._space_loop_limits[spc_var]
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

        for spc_var, block_size in reversed(zip(list(self.space_dims), self.block_sizes)):
            orig_var = str(self._var_map[spc_var])
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
        for spc_var, block_size in reversed(zip(list(self.space_dims), self.block_sizes)):
            # if block size set to None do not block this dimension
            if block_size is not None:
                orig_var = str(self._var_map[spc_var])
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
        for i in range(0, remainder_counter):
            remainder_loop = orig_loop_body
            inner_most_dim = True

            for spc_var, block_size in reversed(zip(list(self.space_dims),
                                                    self.block_sizes)):
                orig_var = str(self._var_map[spc_var])
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

        # replace 0 values with optimal block sizes
        opt_block_size = get_optimal_block_size(self.shape, self.get_number_of_loads())
        for i in range(0, len(self.block_sizes)):
            if self.block_sizes[i] is not None:  # replace 0 values with optimal
                self.block_sizes[i] = (opt_block_size if self.block_sizes[i] == 0
                                       else self.block_sizes[i])
                self.fd.add_value_param(str(self.loop_counters[i]) + "block", np.int64)

    def add_inner_most_dim_pragma(self, inner_most_dim, space_dims, loop_body):
        """
        Adds pragma to inner most dim
        :param inner_most_dim: flag indicating whether its inner most dim
        :param space_dims: space dimensions of kernel
        :param loop_body: original loop body
        :return: cgen.Block - loop body with pragma
        """
        if inner_most_dim and len(space_dims) > 1:
            pragma = [self.get_aligned_pragma(self.sub_stencils, self.factorized,
                                              self.loop_counters, self.time_steppers)]\
                if self.compiler.openmp else (self.compiler.pragma_ivdep +
                                              self.compiler.pragma_nontemporal)
            loop_body = cgen.Block(pragma + [loop_body])
        return loop_body

    def _decide_weights(self, block_sizes, remainder_counter):
        """
        Decided weights which are used for remainder loop limit calculations
        :param block_sizes: list of block sizes
        :param remainder_counter: int stating how many remainder loops are needed
        :return: dict of weights
        """
        weights = {'i3': 0, 'i2': 0, 'i1': 0}
        if len(block_sizes) == 3 and remainder_counter > 1:
            if block_sizes[0] and block_sizes[1] and block_sizes[2]:
                weights.update({'i1': -1})
                if remainder_counter == 3:
                    weights.update({'i2': -2})
            elif (block_sizes[0] and block_sizes[1] and not block_sizes[2]) or\
                 (not block_sizes[0] and block_sizes[1] and block_sizes[2]):
                weights.update({'i2': -1})
            else:
                weights.update({'i1': -1})

        elif (len(block_sizes) == 2 and remainder_counter > 1 and
              block_sizes[0] and block_sizes[1]):
            weights.update({'i1': -1})

        return weights

    def add_loop_step(self, assign, before=False):
        """Add loop step to loop body"""
        stm = self.convert_equality_to_cgen(assign)

        if before:
            self._pre_kernel_steps.append(stm)
        else:
            self._post_kernel_steps.append(stm)

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
        try:  # Assume we have been given a a loop body in cgen types
            self.fd.set_body(self.generate_loops(self.loop_body))
        except:  # We might have been given Sympy expression to evaluate
            # This is the more common use case so this will show up in error messages
            self.fd.set_body(self.generate_loops(self.sympy_to_cgen(self.stencils)))

        return self.fd

    def get_time_stepping(self):
        """Add the time stepping code to the loop

        :returns: A list of :class:`cgen.Statement` containing the time stepping code
        """
        ti = self._var_map[self.time_dim]
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
                array_term = arg

                if not str(array_term.base.label) in self.save_vars:
                    raise ValueError(
                        "Invalid variable '%s' in sympy expression."
                        " Did you add it to the operator's params?"
                        % str(array_term.base.label)
                    )

                if not self.save_vars[str(array_term.base.label)]:
                    subs_dict[arg] = array_term.xreplace(self.t_replace)

        return sympy_expr.xreplace(subs_dict)

    def add_time_loop_stencil(self, stencil, before=False):
        """Add a statement either before or after the main spatial loop, but still inside
        the time loop.

        :param stencil: Given stencil
        :param before: Flag indicating whether the statement should be inserted before,
                       False by default
        """
        if before:
            self.time_loop_stencils_b.append(stencil)
        else:
            self.time_loop_stencils_a.append(stencil)

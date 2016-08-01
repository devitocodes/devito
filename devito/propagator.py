from collections import Iterable
from hashlib import sha1
from os import path
from random import randint

import numpy as np
from sympy import Indexed, IndexedBase, symbols
from sympy.abc import t, x, y, z
from sympy.utilities.iterables import postorder_traversal

import cgen_wrapper as cgen
from codeprinter import ccode
from devito.compiler import (IntelMICCompiler, get_compiler_from_env,
                             get_tmp_dir, jit_compile_and_load)
from devito.function_manager import FunctionDescriptor, FunctionManager


class Propagator(object):
    """Propagator objects derive and encode C kernel code according
    to a given set of stencils, variables and time-stepping options

    :param name: Name of the propagator kernel
    :param nt: Number of timesteps to execute
    :param shape: Shape of the data buffer over which to execute
    :param stencils: List of :class:`sympy.Eq` used to create the kernel
    :param factorized: A map given by {string_name:sympy_object} for including factorized terms
    :param spc_border: Number of spatial padding layers
    :param time_order: Order of the time discretisation
    :param time_dim: Symbol that defines the time dimension
    :param space_dims: List of symbols that define the space dimensions
    :param forward: Flag indicating whether to execute forward in time
    :param compiler: Compiler class used to perform JIT compilation.
                     If not provided, the compiler will be inferred from the
                     environment variable DEVITO_ARCH, or default to GNUCompiler
    :param profile: Flag to enable performance profiling
    :param cache_blocking: Flag to enable cache blocking
    :param block_size: Block size used for cache clocking. Can be either a single number used for all dimensions or
                      a list stating block sizes for each dimension. Set block size to None to skip blocking on that dim
    """

    def __init__(self, name, nt, shape, stencils, factorized=None, spc_border=0, time_order=0,
                 time_dim=None, space_dims=None, dtype=np.float32, forward=True, compiler=None,
                 profile=False, cache_blocking=False, block_size=5):
        self.stencils = stencils
        self.dtype = dtype
        self.factorized = factorized or []
        self.time_order = time_order

        # Default time and space symbols if not provided
        self.time_dim = time_dim or t
        self.space_dims = space_dims or (x, z) if len(shape) == 2 else (x, y, z)[:len(shape)]

        # Internal flags and meta-data
        self.loop_counters = symbols("i1 i2 i3 i4")
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

        # Settings for performance profiling
        self.profile = profile

        if self.profile:
            self.add_local_var("time", "double")
            self.pre_loop.append(cgen.Statement("struct timeval start, end"))
            self.pre_loop.append(cgen.Assign("time", 0))
            self.post_loop.append(cgen.PrintStatement("time: %f\n", "time"))

        # Kernel operational intensity dictionary
        self._kernel_dic_oi = {'add': 0, 'mul': 0, 'load': 0, 'store': 0,
                               'load_list': [], 'load_all_list': [], 'oi_high': 0,
                               'oi_high_weighted': 0, 'oi_low': 0, 'oi_low_weighted': 0}

        # Cache blocking and default block sizes
        self.cache_blocking = cache_blocking

        if isinstance(block_size, Iterable):
            if len(block_size) == len(shape):
                self.block_sizes = block_size
            else:
                raise ValueError("Block size should either be a single number or" +
                                 " an array of the same size as the spatial domain")
        elif block_size is None:  # Turn off cache blocking if block size set to None
            self.cache_blocking = False
        else:
            # A single block size has been passed. Broadcast it to a list of the size of shape
            self.block_sizes = [int(block_size)]*len(shape)

        # Cache C code, lib and function objects
        self._ccode = None
        self._lib = None
        self._cfunction = None

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

    def prep_variable_map(self):
        """Mapping from model variables (x, y, z, t) to loop variables (i1, i2, i3, i4)

        For now, i1 i2 i3 are assigned in the order the variables were defined in init( #var_order)
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
                if self.dtype is np.float32:
                    factors.append(cgen.Assign(name, str(ccode(expr.xreplace(self._var_map))).
                                               replace("pow", "powf").replace("fabs", "fabsf")))
                else:
                    factors.append(cgen.Assign(name, str(ccode(expr.xreplace(self._var_map)))))
        stmts = []

        for equality in stencils:
            self._kernel_dic_oi = self._get_ops_expr(equality.rhs, self._kernel_dic_oi, False)
            self._kernel_dic_oi = self._get_ops_expr(equality.lhs, self._kernel_dic_oi, True)
            stencil = self.convert_equality_to_cgen(equality)
            stmts.append(stencil)

        kernel = self._pre_kernel_steps
        kernel += stmts
        kernel += self._post_kernel_steps

        return cgen.Block(factors+kernel)

    def convert_equality_to_cgen(self, equality):
        """Convert given equality to :class:`cgen.Generable` statement

        :param equality: Given equality statement
        :returns: The resulting :class:`cgen.Generable` statement
        """
        if isinstance(equality, cgen.Generable):
            return equality
        else:
            s_lhs = ccode(self.time_substitutions(equality.lhs).xreplace(self._var_map))
            s_rhs = ccode(self.time_substitutions(equality.rhs).xreplace(self._var_map))
            if self.dtype is np.float32:
                s_rhs = str(s_rhs).replace("pow", "powf")
                s_rhs = str(s_rhs).replace("fabs", "fabsf")
            return cgen.Assign(s_lhs, s_rhs)

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
        if self.cache_blocking:
            loop_body = self.generate_space_loops_blocking(loop_body)
        else:
            loop_body = self.generate_space_loops(loop_body)

        omp_master = [cgen.Pragma("omp master")] if self.compiler.openmp else []
        omp_single = [cgen.Pragma("omp single")] if self.compiler.openmp else []
        omp_parallel = [cgen.Pragma("omp parallel")] if self.compiler.openmp else []
        omp_for = [cgen.Pragma("omp for schedule(static)")] if self.compiler.openmp else []
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
        time_loop_stencils_b = [self.convert_equality_to_cgen(x) for x in self.time_loop_stencils_b]

        # Statements to be inserted into the time loop after the spatial loop
        time_loop_stencils_a = [self.convert_equality_to_cgen(x) for x in self.time_loop_stencils_a]

        if self.profile:
            start_time_stmt = omp_master + [cgen.Block([cgen.Statement("gettimeofday(&start, NULL)")])]
            end_time_stmt = omp_master + [cgen.Block(
                [cgen.Statement("gettimeofday(&end, NULL)")] +
                [cgen.Statement("time += ((end.tv_sec - start.tv_sec) * ",
                                "1000000u + end.tv_usec - start.tv_usec) / 1.e6")]
            )]
        else:
            start_time_stmt = []
            end_time_stmt = []

        initial_block = omp_single + ([cgen.Block(time_stepping + time_loop_stencils_b)]
                                      if time_stepping or time_loop_stencils_b else [])
        initial_block = initial_block + start_time_stmt
        end_block = end_time_stmt + omp_single + ([cgen.Block(time_loop_stencils_a)]
                                                  if time_loop_stencils_a else end_time_stmt)
        loop_body = cgen.Block(initial_block + loop_body + end_block)
        loop_body = cgen.For(
            cgen.InlineInitializer(cgen.Value("int", t_var), str(t_loop_limits[0])),
            t_var + cond_op + str(t_loop_limits[1]),
            t_var + "+=" + str(self._time_step),
            loop_body
        )

        # Code to declare the time stepping variables (outside the time loop)
        def_time_step = [cgen.Value("int", t_var_def.name) for t_var_def in self.time_steppers]
        body = def_time_step + self.pre_loop + omp_parallel + [loop_body] + self.post_loop

        return cgen.Block(body)

    def generate_space_loops(self, loop_body):
        """Generate list<cgen.For> for a non cache blocking space loop
        :param loop_body: Statement representing the loop body
        :returns: :list<cgen.For> a list of for loops
        """
        ivdep = True

        for spc_var in reversed(list(self.space_dims)):
            dim_var = self._var_map[spc_var]
            loop_limits = self._space_loop_limits[spc_var]
            loop_body = cgen.For(
                cgen.InlineInitializer(cgen.Value("int", dim_var), str(loop_limits[0])),
                str(dim_var) + "<" + str(loop_limits[1]),
                str(dim_var) + "++",
                loop_body
            )

            if ivdep and len(self.space_dims) > 1:
                loop_body = cgen.Block(self.compiler.pragma_ivdep + self.compiler.pragma_nontemporal + [loop_body])
            ivdep = False
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
                lower_limit_str = block_var
                upper_limit_str = block_var + "+" + str(block_size)
            else:
                lower_limit_str = str(loop_limits[0])
                upper_limit_str = str(loop_limits[1])

            loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", orig_var), lower_limit_str),
                                 orig_var + "<" + upper_limit_str, orig_var + "++", loop_body)

            if inner_most_dim and len(self.space_dims) > 1:
                loop_body = cgen.Block(self.compiler.pragma_ivdep + self.compiler.pragma_nontemporal + [loop_body])
            inner_most_dim = False

        remainder_counter = 0  # indicates how many remainder loops we need
        for spc_var, block_size in reversed(zip(list(self.space_dims), self.block_sizes)):
            # if block size set to None do not block this dimension
            if block_size is not None:
                orig_var = str(self._var_map[spc_var])
                block_var = orig_var + "b"
                loop_limits = self._space_loop_limits[spc_var]
                old_upper_limit = loop_limits[1]                  # sets new upper limit
                loop_limits = (loop_limits[0], loop_limits[1] - (loop_limits[1] - loop_limits[0]) % block_size)

                if old_upper_limit - loop_limits[1] > 0:  # check old vs new upper
                    remainder_counter += 1

                loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", block_var), str(loop_limits[0])),
                                     str(block_var) + "<" + str(loop_limits[1]), str(block_var) + "+=" +
                                     str(block_size), loop_body)

        full_remainder = []
        weights = self._decide_weights(self.block_sizes, remainder_counter)  # weights for deciding remainder loop limit
        for i in range(0, remainder_counter):
            remainder_loop = orig_loop_body
            inner_most_dim = True

            for spc_var, block_size in reversed(zip(list(self.space_dims), self.block_sizes)):
                orig_var = str(self._var_map[spc_var])
                loop_limits = self._space_loop_limits[spc_var]  # Full loop limits

                if block_size is not None:
                    if weights[orig_var] < 0:
                        # already blocked loop limits
                        loop_limits = (loop_limits[0], loop_limits[1] - (loop_limits[1] - loop_limits[0]) % block_size)
                    elif weights[orig_var] == 0:
                        # remainder loop limits
                        loop_limits = (loop_limits[1] - (loop_limits[1] - loop_limits[0]) % block_size, loop_limits[1])

                    weights[orig_var] += 1

                    # If loop limits are equal that means no remainder on that dim, thus we want all iteration space
                    if loop_limits[0] == loop_limits[1]:
                        loop_limits = self._space_loop_limits[spc_var]

                remainder_loop = cgen.For(cgen.InlineInitializer(cgen.Value("int", orig_var), str(loop_limits[0])),
                                          str(orig_var) + "<" + str(loop_limits[1]), str(orig_var) + "++",
                                          remainder_loop)

                if inner_most_dim and len(self.space_dims) > 1:
                    remainder_loop = cgen.Block(self.compiler.pragma_ivdep + [remainder_loop])

                inner_most_dim = False
            full_remainder.append(remainder_loop)

        return [loop_body] + full_remainder if full_remainder else [loop_body]

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

        elif len(block_sizes) == 2 and block_sizes[0] and block_sizes[1] and remainder_counter > 1:
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
        in the format that FunctionManager and JitManager can deal with it. Before calling,
        make sure you have either called set_jit_params or set_jit_simple already

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
                rhs = ccode((self.time_steppers[i+step_backwards]+1) % (self.time_order+1))

            body.append(cgen.Assign(lhs, rhs))

        return body

    def time_substitutions(self, sympy_expr):
        """This method checks through the sympy_expr to replace the time index with a cyclic index
        but only for variables which are not being saved in the time domain

        :param sympy_expr: The Sympy expression to process
        :returns: The expression after the substitutions
        """
        subs_dict = {}

        for arg in postorder_traversal(sympy_expr):
            if isinstance(arg, Indexed):
                array_term = arg

                if not str(array_term.base.label) in self.save_vars:
                    raise ValueError(
                        "Invalid variable '%s' in sympy expression. Did you add it to the operator's params?"
                        % str(array_term.base.label)
                    )

                if not self.save_vars[str(array_term.base.label)]:
                    subs_dict[arg] = array_term.xreplace(self.t_replace)

        return sympy_expr.xreplace(subs_dict)

    def add_time_loop_stencil(self, stencil, before=False):
        """Add a statement either before or after the main spatial loop, but still inside the time loop.

        :param stencil: Given stencil
        :param before: Flag indicating whether the statement should be inserted before, False by default
        """
        if before:
            self.time_loop_stencils_b.append(stencil)
        else:
            self.time_loop_stencils_a.append(stencil)

    def _get_ops_expr(self, expr, dict1, is_lhs=False):
        """Get number of different operations in expression expr

        Types of operations are ADD (inc -) and MUL (inc /), arrays (IndexedBase objects) in expr that are not in list
        arrays are added to the list.

        :param expr: The expression to process
        :returns: Dictionary of (#ADD, #MUL, list of unique names of fields, list of unique field elements)
        """
        result = dict1  # dictionary to return

        # add array to list arrays if it is not in it
        if isinstance(expr, Indexed):
                base = expr.base.label

                if is_lhs:
                        result['store'] += 1
                if base not in result['load_list']:
                        result['load_list'] += [base]  # accumulate distinct array
                if expr not in result['load_all_list']:
                        result['load_all_list'] += [expr]  # accumulate distinct array elements

                return result

        if expr.is_Mul or expr.is_Add or expr.is_Pow:
                args = expr.args

                # increment MUL or ADD by # arguments less 1
                # sympy multiplication and addition can have multiple arguments
                if expr.is_Mul:
                        result['mul'] += len(args)-1
                else:
                        if expr.is_Add:
                                result['add'] += len(args)-1

                # recursive call of all arguments
                for expr2 in args:
                        result2 = self._get_ops_expr(expr2, result, is_lhs)

                return result2

        # return zero and unchanged array if execution gets here
        return result

    def get_kernel_oi(self, dtype=np.float32):
        """Get the operation intensity of the kernel. The types of operations are ADD (inc -), MUL (inc /), LOAD, STORE.
        #LOAD = number of unique fields in the kernel

        Operation intensity OI = (ADD+MUL)/[(LOAD+STORE)*word_size]
        Weighted OI, OI_w = (ADD+MUL)/(2*Max(ADD,MUL)) * OI

        :param dtype: :class:`numpy.dtype` used to specify the word size
        :returns: A tuple (#ADD, #MUL, #LOAD, #STORE) containing the operation intensity
        """
        load = 0
        load_all = 0
        word_size = np.dtype(dtype).itemsize
        load += len(self._kernel_dic_oi['load_list'])
        store = self._kernel_dic_oi['store']
        load_all += len(self._kernel_dic_oi['load_all_list'])
        self._kernel_dic_oi['load'] = load_all
        add = self._kernel_dic_oi['add']
        mul = self._kernel_dic_oi['mul']
        self._kernel_dic_oi['oi_high'] = float(add+mul)/(load+store)/word_size
        self._kernel_dic_oi['oi_high_weighted'] = self._kernel_dic_oi['oi_high']*(add+mul)/max(add, mul)/2.0
        self._kernel_dic_oi['oi_low'] = float(add+mul)/(load_all+store)/word_size
        self._kernel_dic_oi['oi_low_weighted'] = self._kernel_dic_oi['oi_low']*(add+mul)/max(add, mul)/2.0

        return self._kernel_dic_oi['oi_high']

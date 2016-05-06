import cgen_wrapper as cgen
from codeprinter import ccode
import numpy as np
from sympy import symbols, IndexedBase, Indexed
from function_manager import FunctionDescriptor


class Propagator(object):
    def __init__(self, name, nt, shape, spc_border=0, forward=True, time_order=0):
        num_spac_dim = len(shape)
        self.t = symbols("t")
        space_dims = symbols("x y z")
        self.loop_counters = symbols("i1 i2 i3 i4")
        self._pre_kernel_steps = []
        self._post_kernel_steps = []
        self._forward = forward
        self.space_dims = space_dims[0:num_spac_dim]
        self.prep_variable_map()
        self.t_replace = {}
        self.time_steppers = []
        self.time_order = time_order
        self.nt = nt
        self.time_loop_stencils_b = []
        self.time_loop_stencils_a = []
        # Start with the assumption that the propagator needs to save the field in memory at every time step
        self._save = True
        # This might be changed later when parameters are being set

        # Which function parameters need special (non-save) time stepping and which don't
        self.save_vars = {}
        self.fd = FunctionDescriptor(name)
        if forward:
            self._time_step = 1
        else:
            self._time_step = -1
        self._space_loop_limits = {}
        for i, dim in enumerate(reversed(self.space_dims)):
                self._space_loop_limits[dim] = (spc_border, shape[i]-spc_border)

        # Kernel ai dictionary
        self._kernel_dic_ai = {'add': 0, 'mul': 0, 'load': 0, 'store': 0, 'load_list': [], 'load_all_list': [], 'ai_high': 0, 'ai_high_weighted': 0, 'ai_low': 0, 'ai_low_weighted': 0}
        self._print_ai = False
        self._dtype = False

    @property
    def save(self):
        return self._save

    @save.setter
    def save(self, save):
        if save is not True:
            self.time_steppers = [symbols("t%d" % i) for i in range(self.time_order+1)]
            if self._forward is not True:
                self.t_replace = {(self.t): self.time_steppers[2], (self.t+1): self.time_steppers[1], (self.t+2): self.time_steppers[0]}
            else:
                self.t_replace = {(self.t): self.time_steppers[2], (self.t-1): self.time_steppers[1], (self.t-2): self.time_steppers[0]}
        self._save = self._save and save

    @property
    def time_loop_limits(self):
        num_save_vars = sum([1 for x in self.save_vars.values() if x is True])
        skip_time = self.time_order if num_save_vars > 0 else 0
        if self._forward:
            loop_limits = (0+skip_time, self.nt+skip_time)
        else:
            loop_limits = (self.nt-1, -1)
        return loop_limits

    def prep_variable_map(self):
        """ Mapping from model variables (x, y, z, t) to loop variables (i1, i2, i3, i4) - Needs work
        """
        var_map = {}
        i = 0
        for dim in self.space_dims:
            var_map[dim] = symbols("i%d" % (i + 1))
            i += 1
        var_map[self.t] = symbols("i%d" % (i + 1))
        self._var_map = var_map

    def prepare(self, subs, stencils, stencil_args):
        stmts = []
        for equality, args in zip(stencils, stencil_args):
            equality = equality.subs(dict(zip(subs, args)))
            self._kernel_dic_ai = self._get_ops_expr(equality.rhs, self._kernel_dic_ai, False)
            self._kernel_dic_ai = self._get_ops_expr(equality.lhs, self._kernel_dic_ai, True)
            stencil = self.convert_equality_to_cgen(equality)
            stmts.append(stencil)
        kernel = self._pre_kernel_steps
        kernel += stmts
        kernel += self._post_kernel_steps
        if self._print_ai:
                print(self._get_kernel_ai(self._dtype))
        return self.prepare_loop(cgen.Block(kernel))

    def convert_equality_to_cgen(self, equality):
        return cgen.Assign(ccode(self.time_substitutions(equality.lhs).xreplace(self._var_map)), ccode(self.time_substitutions(equality.rhs).xreplace(self._var_map)))

    def prepare_loop(self, loop_body):
        num_spac_dim = len(self.space_dims)
        for dim_ind in range(1, num_spac_dim+1):
            dim_var = "i"+str(dim_ind)
            loop_limits = self._space_loop_limits[self.space_dims[dim_ind-1]]
            loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var),
                                                        str(loop_limits[0])),
                                 dim_var + "<" + str(loop_limits[1]), dim_var + "++", loop_body)
        t_loop_limits = self.time_loop_limits
        t_var = str(self._var_map[self.t])
        cond_op = "<" if self._forward else ">"
        if self.save is not True:
            time_stepping = self.get_time_stepping()
        else:
            time_stepping = []
        time_loop_stencils_b = [self.convert_equality_to_cgen(x) for x in self.time_loop_stencils_b]
        time_loop_stencils_a = [self.convert_equality_to_cgen(x) for x in self.time_loop_stencils_a]
        loop_body = cgen.Block(time_stepping + time_loop_stencils_b + [loop_body] + time_loop_stencils_a)
        loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", t_var), str(t_loop_limits[0])), t_var + cond_op + str(t_loop_limits[1]), t_var + "+=" + str(self._time_step), loop_body)
        def_time_step = [cgen.Value("int", t_var_def.name) for t_var_def in self.time_steppers]
        body = def_time_step + [loop_body]
        return cgen.Block(body)

    def add_loop_step(self, assign, before=False):
        self._kernel_dic_ai = self._get_ops_expr((self.time_substitutions(assign.lhs).xreplace(self._var_map)), self._kernel_dic_ai, True)
        if isinstance(assign.rhs, Indexed):
                self._kernel_dic_ai = self._get_ops_expr((self.time_substitutions(assign.rhs).xreplace(self._var_map)), self._kernel_dic_ai, False)
        stm = self.convert_equality_to_cgen(assign)
        if before:
            self._pre_kernel_steps.append(stm)
        else:
            self._post_kernel_steps.append(stm)

    def set_jit_params(self, subs, stencils, stencil_args):
        self.subs = subs
        self.stencils = stencils
        self.stencil_args = stencil_args

    def set_jit_simple(self, loop_body):
        self.loop_body = loop_body

    def add_param(self, name, shape, dtype, save=True):
        self.fd.add_matrix_param(name, shape, dtype)
        self.save = save
        self.save_vars[name] = save
        return IndexedBase(name, shape=shape)

    def add_scalar_param(self, name, dtype):
        self.fd.add_value_param(name, dtype)
        return symbols(name)

    def add_local_var(self, name, dtype):
        self.fd.add_local_variable(name, dtype)
        return symbols(name)

    def get_fd(self):
        """Get a FunctionDescriptor that describes the code represented by this Propagator
        in the format that FunctionManager and JitManager can deal with it. Before calling,
        make sure you have either called set_jit_params or set_jit_simple already.
        """
        try:  # Assume we have been given a a loop body in cgen types
            self.fd.set_body(self.prepare_loop(self.loop_body))
        except:  # We might have been given Sympy expression to evaluate
            # This is the more common use case so this will show up in error messages
            self.fd.set_body(self.prepare(self.subs, self.stencils, self.stencil_args))
        return self.fd

    def get_time_stepping(self):
        ti = self._var_map[self.t]
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
        """
        if isinstance(sympy_expr, Indexed):
            array_term = sympy_expr
            if not self.save_vars[str(array_term.base.label)]:
                array_term = array_term.xreplace(self.t_replace)
            return array_term
        else:
            for arg in sympy_expr.args:
                sympy_expr = sympy_expr.subs(arg, self.time_substitutions(arg))
        return sympy_expr

    def add_time_loop_stencil(self, stencil, before=False):
        self._kernel_dic_ai = self._get_ops_expr(stencil.lhs, self._kernel_dic_ai, True)
        self._kernel_dic_ai = self._get_ops_expr(stencil.rhs, self._kernel_dic_ai, False)
        if before:
            self.time_loop_stencils_b.append(stencil)
        else:
            self.time_loop_stencils_a.append(stencil)

    def enable_ai(self, is_enable=False, dtype=None):
        """Update variable to enable propagation print its kernel ai
        with its proper type.
        """
        self._print_ai = is_enable
        self._dtype = dtype

    def _get_ops_expr(self, expr, dict1, is_lhs=False):
        """
        - get number of different operations in expression expr
        - types of operations are ADD (inc -) and MUL (inc /)
        - arrays (IndexedBase objects) in expr that are not in list arrays
        are added to the list
        - return dictionary of (#ADD, #MUL, list of unique names of fields, list of unique field elements)
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

    def _get_kernel_ai(self, dtype=None):
        """
        - get the arithmetic intensity of the kernel
        - types of operations are ADD (inc -), MUL (inc /), LOAD, STORE
        - #LOAD = number of unique fields in the kernel
        - return tuple (#ADD, #MUL, #LOAD, #STORE)
        - arithmetic intensity AI = (ADD+MUL)/[(LOAD+STORE)*word size]
        - weighted AI, AI_w = (ADD+MUL)/(2*Max(ADD,MUL)) * AI
        """
        load = 0
        load_all = 0
        word_size = np.dtype(dtype).itemsize if dtype is not None else 8
        load += len(self._kernel_dic_ai['load_list'])
        store = self._kernel_dic_ai['store']
        load_all += len(self._kernel_dic_ai['load_all_list'])
        self._kernel_dic_ai['load'] = load
        add = self._kernel_dic_ai['add']
        mul = self._kernel_dic_ai['mul']
        self._kernel_dic_ai['ai_high'] = float(add+mul)/(load+store)/word_size
        self._kernel_dic_ai['ai_high_weighted'] = self._kernel_dic_ai['ai_high']*(add+mul)/max(add, mul)/2.0
        self._kernel_dic_ai['ai_low'] = float(add+mul)/(load_all+store)/word_size
        self._kernel_dic_ai['ai_low_weighted'] = self._kernel_dic_ai['ai_low']*(add+mul)/max(add, mul)/2.0

        return self._kernel_dic_ai

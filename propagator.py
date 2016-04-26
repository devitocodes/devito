import cgen_wrapper as cgen
from codeprinter import ccode
from sympy import symbols


class Propagator:
    def __init__(self, nt, shape, spc_border=0, forward=True, skip_time = 0, save = True):
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
        self.save = save
        self.time_order = 2 # Hard-coded since the equation we are solving now is second-order in time
        # The code is expected (but not tested) to work with other time orders as well
        if forward:
            self._loop_limits = {self.t: (0+skip_time, nt+skip_time)}
            self._time_step = 1
        else:
            self._loop_limits = {self.t: (nt-1, -1)}
            self._time_step = -1
            if save is not True:
                self.time_steppers = [symbols("t%d" % i) for i in range(self.time_order+1)]
                self.t_replace = {(self.t): self.time_steppers[2], (self.t+1): self.time_steppers[1], (self.t+2): self.time_steppers[0]}
        for i, dim in enumerate(reversed(self.space_dims)):
                self._loop_limits[dim] = (spc_border, shape[i]-spc_border)

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

    def prepare(self, subs, stencils, stencil_args, debug = False):
        stmts = []
        for equality in stencils:
            equality = equality.subs(dict(zip(subs, stencil_args))).xreplace(self.t_replace)
            equality = equality.xreplace(self._var_map)
            stencil = cgen.Assign(ccode(equality.lhs), ccode(equality.rhs))
            stmts.append(stencil)
        kernel = self._pre_kernel_steps
        kernel += stmts
        kernel += self._post_kernel_steps
        return self.prepare_loop(cgen.Block(kernel), debug)

    def prepare_loop(self, loop_body, debug = False):
        num_spac_dim = len(self.space_dims)
        for dim_ind in range(1, num_spac_dim+1):
            dim_var = "i"+str(dim_ind)
            loop_limits = self._loop_limits[self.space_dims[dim_ind-1]]
            loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var),
                                                        str(loop_limits[0])),
                                 dim_var + "<" + str(loop_limits[1]), dim_var + "++", loop_body)
        t_loop_limits = self._loop_limits[self.t]
        t_var = str(self._var_map[self.t])
        cond_op = "<" if self._forward else ">"
        if debug:
            print_st = [cgen.Statement("printf(\"%f, %f, %f, %f\\n\", v3[50][50], v2[50][50], v1[50][50], u[i3+2][50][50])")]
        else:
            print_st = []
        if self.save is not True:
            time_stepping = self.get_time_stepping()
        else:
            time_stepping = []
        loop_body = cgen.Block(time_stepping+ [loop_body] + print_st)
        loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", t_var), str(t_loop_limits[0])), t_var + cond_op + str(t_loop_limits[1]), t_var + "+=" + str(self._time_step), loop_body)
        def_time_step = [cgen.Value("int", t_var.name) for t_var in self.time_steppers]
        body = def_time_step + [loop_body]
        return cgen.Block(body)

    def add_loop_step(self, sympy_condition, true_assign, false_assign=None, before=False):
        condition = ccode(sympy_condition.lhs.xreplace(self._var_map)) + " == " + ccode(sympy_condition.rhs.xreplace(self._var_map))
        true_str = cgen.Assign(ccode(true_assign.lhs.xreplace(self._var_map)), ccode(true_assign.rhs.xreplace(self._var_map)))
        false_str = cgen.Assign(ccode(false_assign.lhs.xreplace(self._var_map)), ccode(false_assign.rhs.xreplace(self._var_map))) if false_assign is not None else None
        statement = cgen.If(condition, true_str, false_str)
        if before:
            self._pre_kernel_steps.append(statement)
        else:
            self._post_kernel_steps.append(statement)

    def get_time_stepping(self):
        ti = self._var_map[self.t]
        body = []
        for i in range(self.time_order+1):
            lhs = self.time_steppers[i].name
            if i == 0:
                rhs = ccode(ti % (self.time_order+1))
            else:
                rhs = ccode((self.time_steppers[i-1]+1) % (self.time_order+1))
            body.append(cgen.Assign(lhs, rhs))

        return body

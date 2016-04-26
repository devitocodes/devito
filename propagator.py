import cgen_wrapper as cgen
from codeprinter import ccode
from sympy import symbols


class Propagator:
    def __init__(self, nt, shape, spc_border=0, forward=True, skip_time = 0):
        num_spac_dim = len(shape)
        self.t = symbols("t")
        space_dims = symbols("x y z")
        self.loop_counters = symbols("i1 i2 i3")
        self._pre_kernel_steps = []
        self._post_kernel_steps = []
        self._time_loop_extras = []
        self._forward = forward
        self.space_dims = space_dims[0:num_spac_dim]
        self._prep_loop_order()
        if forward:
            self._loop_limits = {self.t: (0+skip_time, nt)}
            self._time_step = 1
        else:
            self._loop_limits = {self.t: (nt-1, -1)}
            self._time_step = -1
        for i, dim in enumerate(reversed(self.space_dims)):
                self._loop_limits[dim] = (spc_border, shape[i]-spc_border)

    def _prep_loop_order(self):
        """ Mapping from model variables (x, y, z, t) to loop variables (i1, i2, i3, i4) - Needs work
        """
        loop_order = {}
        i = 0
        for dim in self.space_dims:
            loop_order[dim] = symbols("i%d" % (i + 1))
            i += 1
        loop_order[self.t] = symbols("i%d" % (i + 1))
        self._loop_order = loop_order

    def prepare(self, subs, stencils, stencil_args):
        stmts = []
        for equality in stencils:
            equality = equality.subs(dict(zip(subs, stencil_args)))
            equality = equality.xreplace(self._loop_order)
            stencil = cgen.Assign(ccode(equality.lhs), ccode(equality.rhs))
            stmts.append(stencil)
        kernel = self._pre_kernel_steps
        kernel += stmts
        kernel += self._post_kernel_steps
        return self.prepare_loop(cgen.Block(kernel))

    def prepare_loop(self, loop_body):
        num_spac_dim = len(self.space_dims)
        for dim_ind in range(1, num_spac_dim+1):
            dim_var = "i"+str(dim_ind)
            loop_limits = self._loop_limits[self.space_dims[dim_ind-1]]
            loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var),
                                                        str(loop_limits[0])),
                                 dim_var + "<" + str(loop_limits[1]), dim_var + "++", loop_body)
        t_loop_limits = self._loop_limits[self.t]
        t_var = str(self._loop_order[self.t])
        cond_op = "<" if self._forward else ">"
        loop_body = cgen.Block([loop_body] + self._time_loop_extras)
        loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", t_var), str(t_loop_limits[0])), t_var + cond_op + str(t_loop_limits[1]), t_var + "+=" + str(self._time_step), loop_body)
        return loop_body

    def add_loop_step(self, sympy_condition, true_assign, false_assign=None, before=False):
        condition = ccode(sympy_condition.lhs.xreplace(self._loop_order)) + " == " + ccode(sympy_condition.rhs.xreplace(self._loop_order))
        true_str = cgen.Assign(ccode(true_assign.lhs.xreplace(self._loop_order)), ccode(true_assign.rhs.xreplace(self._loop_order)))
        false_str = cgen.Assign(ccode(false_assign.lhs.xreplace(self._loop_order)), ccode(false_assign.rhs.xreplace(self._loop_order))) if false_assign is not None else None
        statement = cgen.If(condition, true_str, false_str)
        if before:
            self._pre_kernel_steps.append(statement)
        else:
            self._post_kernel_steps.append(statement)

    def add_time_loop(self, time_steps):
        self._time_loop_extras += [cgen.Assign(ccode(step.lhs.xreplace(self._loop_order)), ccode(step.rhs.xreplace(self._loop_order))) for step in time_steps]

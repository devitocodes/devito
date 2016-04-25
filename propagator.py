import cgen_wrapper as cgen
from codeprinter import ccode
from sympy import symbols


class Propagator:
    def __init__(self, nt, shape, border=0, forward=True):
        num_spac_dim = len(shape)
        self.t = symbols("t")
        space_dims = symbols("x y z")
        self.space_dims = space_dims[0:num_spac_dim]
        self.loop_counters = symbols("i1 i2 i3")
        self._prep_loop_order()
        self._pre_kernel_steps = []
        self._post_kernel_steps = []
        self._forward = forward
        if forward:
            self._loop_limits = {self.t: (0, nt)}
            self._time_step = 1
        else:
            self._loop_limits = {self.t: (nt-1, -1)}
            self._time_step = -1
        for i, dim in enumerate(reversed(self.space_dims)):
                self._loop_limits[dim] = (border, shape[i]-border)

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

    def prepare(self, subs, stencil, stencil1_args, stencil2_args, stencil3_args, lhs):
        stencil1 = self.process_stencil(stencil, subs, stencil1_args, lhs)
        stencil2 = self.process_stencil(stencil, subs, stencil2_args, lhs)
        stencil3 = self.process_stencil(stencil, subs, stencil3_args, lhs)
        kernel = self._pre_kernel_steps
        if_combinations = cgen.make_multiple_ifs([("%s == %d" % (self._loop_order[self.t], self._loop_limits[self.t][0]), stencil1), ("%s == %d" % (self._loop_order[self.t], self._loop_limits[self.t][0]+self._time_step), stencil2), (None, stencil3)], "last")
        kernel.append(if_combinations)
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

    def process_stencil(self, original_stencil, subs, stencil_args, lhs):
        stencil = original_stencil.subs(dict(zip(subs, stencil_args)))
        stencil = stencil.xreplace(self._loop_order)
        stencil = cgen.Assign(ccode(lhs.xreplace(self._loop_order)), ccode(stencil))
        return stencil

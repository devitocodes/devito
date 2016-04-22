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
        if forward:
            self._loop_limits = {self.t: (0, nt)}
            for i, dim in enumerate(reversed(self.space_dims)):
                self._loop_limits[dim] = (border, shape[i]-border)
            self._time_step = 1

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
        kernel = []
        if_combinations = cgen.make_multiple_ifs([("%s == %d" % (self._loop_order[self.t], self._loop_limits[self.t][0]), stencil1), ("%s == %d" % (self._loop_order[self.t], self._loop_limits[self.t][0]+self._time_step), stencil2), (None, stencil3)], "last")
        kernel.append(if_combinations)
        copy_receiver = cgen.Assign("rec[%s][%s-1]" % (self._loop_order[self.t], self._loop_order[self.space_dims[1]]), ccode(lhs.xreplace(self._loop_order)))
        copy_receiver_if = cgen.If("%s == xrec" % self._loop_order[self.space_dims[0]], copy_receiver)
        kernel.append(copy_receiver_if)
        return cgen.Block(kernel)

    def process_stencil(self, original_stencil, subs, stencil_args, lhs):
        stencil = original_stencil.subs(dict(zip(subs, stencil_args)))
        stencil = stencil.xreplace(self._loop_order)
        stencil = cgen.Assign(ccode(lhs.xreplace(self._loop_order)), ccode(stencil))
        return stencil

    def prepare_loop(self, loop_body, looper_matrix_name):
        num_spac_dim = len(self.space_dims)
        for dim_ind in range(1, num_spac_dim+1):
            dim_var = "i"+str(dim_ind)
            loop_limits = self._loop_limits[self.space_dims[dim_ind-1]]
            loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var),
                                                        str(loop_limits[0])),
                                 dim_var + "<" + str(loop_limits[1]), dim_var + "++", loop_body)
        t_loop_limits = self._loop_limits[self.t]
        t_var = str(self._loop_order[self.t])
        loop_body = cgen.For(cgen.InlineInitializer(cgen.Value("int", t_var), str(t_loop_limits[0])), t_var + "<" + str(t_loop_limits[1]), t_var + "+=" + str(self._time_step), loop_body)
        return loop_body

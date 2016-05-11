from devito.operators import *
from sympy.abc import x, y, t
from sympy import Eq, symbols, Matrix
from devito.interfaces import TimeData, DenseData


class ForwardOperator(Operator):
    def __init__(self, subs, stencil, m, src, damp, rec, u):
        assert(m.shape==damp.shape)
        self.input_params = [m, src, damp, rec, u]
        u.pad_time = True
        self.output_params = []
        stencil_args = [u[t - 2, x, y],
                        u[t - 1, x - 1, y],
                        u[t - 1, x, y],
                        u[t - 1, x + 1, y],
                        u[t - 1, x, y - 1],
                        u[t - 1, x, y + 1],
                        m[x, y], src.dt, src.h, damp[x, y]]
        main_stencil = Eq(u[t, x, y], stencil)
        self.stencils = [(main_stencil, stencil_args)]
        src_list = src.add(m, u)
        rec = rec.read(u)
        self.time_loop_stencils_post = src_list+rec
        super(ForwardOperator, self).__init__(subs, src.nt, m.shape, spc_border=1, time_order=2, forward=True, dtype=m.dtype)


class AdjointOperator(Operator):
    def __init__(self, subs, stencil, m, rec, damp, srca):
        assert(m.shape==damp.shape)
        self.input_params = [m, rec, damp, srca]
        v = TimeData("v", m.shape, rec.nt, time_order=2, save=True, dtype=m.dtype)
        self.output_params = [v]
        lhs = v[t, x, y]
        main_stencil = Eq(lhs, stencil)
        stencil_args = [v[t + 2, x, y],
                        v[t + 1, x - 1, y],
                        v[t + 1, x, y],
                        v[t + 1, x + 1, y],
                        v[t + 1, x, y - 1],
                        v[t + 1, x, y + 1],
                        m[x, y], rec.dt, rec.h, damp[x,y]]
        self.stencils = [(main_stencil, stencil_args)]
        rec_list = rec.add(m, v)
        src_list = srca.read(v)
        self.time_loop_stencils_post = rec_list + src_list
        super(AdjointOperator, self).__init__(subs, rec.nt, m.shape, spc_border=1, time_order=2, forward=False, dtype=m.dtype)

class GradientOperator(Operator):
    def __init__(self, subs, stencil, u, m, rec, damp):
        assert(m.shape==damp.shape)
        self.input_params = [u, m, rec, damp]
        v = TimeData("v", m.shape, rec.nt, time_order=2, save=False, dtype=m.dtype)
        grad = DenseData("grad", m.shape, dtype=m.dtype)
        self.output_params = [grad, v]
        lhs = v[t, x, y]
        stencil_args = [v[t + 2, x, y], v[t + 1, x - 1, y], v[t + 1, x, y], v[t + 1, x + 1, y], v[t + 1, x, y - 1], v[t + 1, x, y + 1], 
                        m[x, y], rec.dt, rec.h, damp[x,y]]
        main_stencil = Eq(lhs, lhs + stencil)
        gradient_update = Eq(grad[x, y],
                             grad[x, y] - (v[t, x, y] - 2 * v[t + 1, x, y] + v[t + 2, x, y]) * (u[t, x, y]))
        reset_v = Eq(v[t+2, x, y], 0)
        self.stencils = [(main_stencil, stencil_args), (gradient_update, []),  (reset_v, [])]

        rec_list = rec.add(m, v)
        self.time_loop_stencils_pre = rec_list
        super(GradientOperator, self).__init__(subs, rec.nt, m.shape, spc_border=1, time_order=2, forward=False, dtype=m.dtype)


class BornOperator(Operator):
    def __init__(self, subs, stencil, dm, m, src, damp, rec):
        assert(m.shape==damp.shape)
        self.input_params = [dm, m, src, damp, rec]
        u = TimeData("u", m.shape, src.nt, time_order=2, save=False, dtype=m.dtype)
        U = TimeData("U", m.shape, src.nt, time_order=2, save=False, dtype=m.dtype)
        self.output_params = [u, U]
        dt = src.dt
        h = src.h
        src_list = src.add(m, u)
        rec = rec.read(U)
        self.time_loop_stencils_pre = src_list
        self.time_loop_stencils_post = rec
        first_stencil_args = [u[t-2, x, y],
                              u[t-1, x - 1, y],
                              u[t-1, x, y],
                              u[t-1, x + 1, y],
                              u[t-1, x, y - 1],
                              u[t-1, x, y + 1],
                              m[x, y], dt, h, damp[x,y]]
        first_update = Eq(u[t, x, y], u[t, x, y]+stencil)
        src2 = -(dt**-2)*(u[t, x, y]-2*u[t-1, x, y]+u[t-2, x, y])*dm[x, y]
        second_stencil_args = [U[t-2, x, y],
                               U[t-1, x - 1, y],
                               U[t-1, x, y],
                               U[t-1, x + 1, y],
                               U[t-1, x, y - 1],
                               U[t-1, x, y + 1],
                               m[x, y], dt, h, damp[x,y]]
        second_update = Eq(U[t, x, y], stencil)
        insert_second_source = Eq(U[t, x, y], U[t, x, y]+(dt*dt)/m[x, y]*src2)
        reset_u = Eq(u[t-2, x, y], 0)
        self.stencils = [(first_update, first_stencil_args), (second_update, second_stencil_args), (insert_second_source,[]), (reset_u, [])]
        super(BornOperator, self).__init__(subs, src.nt, m.shape, spc_border=1, time_order=2, forward=True, dtype=m.dtype)

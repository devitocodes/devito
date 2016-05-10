from devito.operators import *
from sympy.abc import x, y, t
from sympy import Eq, symbols, Matrix
from devito.interfaces import TimeData


class ForwardOperator(Operator):
    def __init__(self, subs, stencil, m, src, damp, rec):
        assert(m.shape==damp.shape)
        self.input_params = [m, src, damp, rec]
        u = TimeData("u", m.shape, src.nt, time_order=2, save=True, dtype=m.dtype)
        u.pad_time = True
        self.output_params = [u]
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
    def _prepare(self):
        stencil = self.stencil
        subs = self.subs
        nx, ny = self.shape
        nt = self.nt
        nrec = self.nrec
        propagator = Propagator("Gradient", nt, (nx, ny), spc_border=1, forward=False, time_order=2)
        u = propagator.add_param("u", (nt, nx, ny), self.dtype)
        rec = propagator.add_param("rec", (nt, nrec), self.dtype)
        v = propagator.add_param("v", (nt, nx, ny), self.dtype, save=False)
        grad = propagator.add_param("grad", (nx, ny), self.dtype)
        M = propagator.add_param("m", (nx, ny), self.dtype)
        dampx = propagator.add_param("dampx", (nx,), self.dtype)
        dampy = propagator.add_param("dampy", (nx,), self.dtype)
        dt = self.dt
        h = self.h
        lhs = v[t, x, y]
        stencil_args = [v[t + 2, x, y], v[t + 1, x - 1, y], v[t + 1, x, y], v[t + 1, x + 1, y], v[t + 1, x, y - 1], v[t + 1, x, y + 1], M[x, y], dt, h, dampx[x]+dampy[y]]
        main_stencil = Eq(lhs, lhs + stencil)
        gradient_update = Eq(grad[x, y],
                             grad[x, y] - (v[t, x, y] - 2 * v[t + 1, x, y] + v[t + 2, x, y]) * (u[t+2, x, y]))
        propagator.set_jit_params(subs, [main_stencil, gradient_update], [stencil_args, []])
        propagator.add_loop_step(Eq(v[t+2, x, y], 0), False)
        rec = self.add_rec(rec, M, self.dt, v)
        for sten in rec:
            propagator.add_time_loop_stencil(sten, before=True)
        return propagator


class BornOperator(Operator):
    def _prepare(self):
        stencil = self.stencil
        subs = self.subs
        nt = self.nt
        nx, ny = self.shape
        nrec = self.nrec
        propagator = Propagator("Born", nt, (nx, ny), spc_border=1, time_order=2)
        u = propagator.add_param("u", (nt, nx, ny), self.dtype, save=False)
        U = propagator.add_param("U", (nt, nx, ny), self.dtype, save=False)
        rec = propagator.add_param("rec", (nt, nrec), self.dtype)
        dm = propagator.add_param("dm", (nx, ny), self.dtype)
        M = propagator.add_param("m", (nx, ny), self.dtype)
        dampx = propagator.add_param("dampx", (nx,), self.dtype)
        dampy = propagator.add_param("dampy", (nx,), self.dtype)
        src_time = propagator.add_param("src_time", (nt,), self.dtype)
        src2 = propagator.add_local_var("src2", self.dtype)
        dt = self.dt
        h = self.h
        propagator.add_loop_step(Eq(u[t-2, x, y], 0), False)
        src = self.add_source(src_time, M, self.dt, u)
        for sten in src:
            propagator.add_time_loop_stencil(sten, before=True)
        rec = self.read_rec(rec, U)
        for sten in rec:
            propagator.add_time_loop_stencil(sten, before=False)
        first_stencil_args = [u[t-2, x, y],
                              u[t-1, x - 1, y],
                              u[t-1, x, y],
                              u[t-1, x + 1, y],
                              u[t-1, x, y - 1],
                              u[t-1, x, y + 1],
                              M[x, y], dt, h, dampx[x] + dampy[y]]
        first_update = Eq(u[t, x, y], u[t, x, y]+stencil)
        src2 = -(dt**-2)*(u[t, x, y]-2*u[t-1, x, y]+u[t-2, x, y])*dm[x, y]
        second_stencil_args = [U[t-2, x, y],
                               U[t-1, x - 1, y],
                               U[t-1, x, y],
                               U[t-1, x + 1, y],
                               U[t-1, x, y - 1],
                               U[t-1, x, y + 1],
                               M[x, y], dt, h, dampx[x] + dampy[y]]
        second_update = Eq(U[t, x, y], stencil)
        insert_second_source = Eq(U[t, x, y], U[t, x, y]+(dt*dt)/M[x, y]*src2)
        propagator.set_jit_params(subs, [first_update, second_update, insert_second_source], [first_stencil_args, [], second_stencil_args, []])
        return propagator

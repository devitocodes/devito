from devito.operator import *
from sympy import Eq
from devito.interfaces import TimeData, DenseData
from sympy import Function, symbols, as_finite_diff, Wild
from sympy.abc import x, y, t, z
from sympy import solve
from sympy import *
from sympy.abc import *
from fwi_operators import SourceLike


class SourceLikeTTI(SourceLike):
    def read(self, u, v):
        eqs = []
        for i in range(self.npoints):
            eqs.append(Eq(self[t, i], (self.grid2point(u, self.orig_data[i, :]) + self.grid2point(v, self.orig_data[i, :]))/2))
        return eqs


class TTIOperator(Operator):
    def _init_taylor(self, dim=2, time_order=2, space_order=2):
        # Only dim=2 and dim=3 are supported
        # The acoustic wave equation for the square slowness m and a source q
        # is given in 3D by :
        #
        # \begin{cases} &m \frac{d^2 u(x,t)}{dt^2} - \nabla^2 u(x,t) =q  \\
        # &u(.,0) = 0 \\ &\frac{d u(x,t)}{dt}|_{t=0} = 0 \end{cases}
        #
        # with the zero initial conditons to guarantee unicity of the solution
        # Choose dimension (2 or 3)
        # half width for indexes, goes from -half to half
        p = Function('p')
        r = Function('r')

        s, h, x, y, z = symbols('s h x y z')
        m = M(x, y, z)
        # q = Q(x, y, z, t)
        # d = D(x, y, z, t)
        e = E(x, y, z)

        A = epsilon(x, y, z)  # (1 + 2epsilon) but make the symbolic simpler
        B = delta(x, y, z)  # sqrt(1 + 2epsilon) but make the symbolic simpler
        Th = theta(x, y, z)
        Ph = phi(x, y, z)
        # Weights to sum the two fields
        # w1 = .5
        # w2 = .5
        dttp = as_finite_diff(p(x, y, z, t).diff(t, t), [t-s, t, t+s])
        dttr = as_finite_diff(r(x, y, z, t).diff(t, t), [t-s, t, t+s])
        dtp = as_finite_diff(p(x, y, z, t).diff(t), [t-s, t])
        dtr = as_finite_diff(r(x, y, z, t).diff(t), [t-s, t])
        # Spacial finite differences can easily be extended to higher order by increasing the list of sampling point in the next expression.
        # Be sure to keep this stencil symmetric and everything else in the notebook will follow.
        dxxp = as_finite_diff(p(x, y, z, t).diff(x, x), [x-h, x, x+h])
        dyyp = as_finite_diff(p(x, y, z, t).diff(y, y), [y-h, y, y+h])
        dzzp = as_finite_diff(p(x, y, z, t).diff(z, z), [z-h, z, z+h])
        dxxr = as_finite_diff(r(x, y, z, t).diff(x, x), [x-h, x, x+h])
        dyyr = as_finite_diff(r(x, y, z, t).diff(y, y), [y-h, y, y+h])
        dzzr = as_finite_diff(r(x, y, z, t).diff(z, z), [z-h, z, z+h])

        # My 4th order stencil for cross derivatives
        dxzp = .5/(h**2)*(-2*p(x, y, z, t) + p(x, y, z+h, t) + p(x, y, z-h, t) - p(x+h, y, z-h, t) + p(x-h, y, z, t) - p(x-h, y, z+h, t) + p(x+h, y, z, t))
        dxzr = .5/(h**2)*(-2*r(x, y, z, t) + r(x, y, z+h, t) + r(x, y, z-h, t) - r(x+h, y, z-h, t) + r(x-h, y, z, t) - r(x-h, y, z+h, t) + r(x+h, y, z, t))
        dxyp = .5/(h**2)*(-2*p(x, y, z, t) + p(x, y+h, z, t) + p(x, y-h, z, t) - p(x+h, y-h, z, t) + p(x-h, y, z, t) - p(x-h, y+h, z, t) + p(x+h, y, z, t))
        dxyr = .5/(h**2)*(-2*r(x, y, z, t) + r(x, y+h, z, t) + r(x, y-h, z, t) - r(x+h, y-h, z, t) + r(x-h, y, z, t) - r(x-h, y+h, z, t) + r(x+h, y, z, t))
        dyzp = .5/(h**2)*(-2*p(x, y, z, t) + p(x, y, z+h, t) + p(x, y, z-h, t) - p(x, y+h, z-h, t) + p(x, y-h, z, t) - p(x, y-h, z+h, t) + p(x, y+h, z, t))
        dyzr = .5/(h**2)*(-2*r(x, y, z, t) + r(x, y, z+h, t) + r(x, y, z-h, t) - r(x, y+h, z-h, t) + r(x, y-h, z, t) - r(x, y-h, z+h, t) + r(x, y+h, z, t))
        Gxxp = cos(Ph)**2 * cos(Th)**2 * dxxp + sin(Ph)**2 * cos(Th)**2 * dyyp + sin(Th)**2 * dzzp + sin(2*Ph) * cos(Th)**2 * dxyp - sin(Ph) * sin(2*Th) * dyzp - cos(Ph) * sin(2*Th) * dxzp
        Gyyp = sin(Th)**2 * dxxp + cos(Ph)**2 * dyyp - sin(2*Ph)**2 * dxyp
        Gzzr = cos(Ph)**2 * sin(Th)**2 * dxxr + sin(Ph)**2 * sin(Th)**2 * dyyr + cos(Th)**2 * dzzr +\
            sin(2*Ph) * sin(Th)**2 * dxyr + sin(Ph) * sin(2*Th) * dyzr + cos(Ph) * sin(2*Th) * dxzr
        wavep = m * dttp - A * (Gxxp + Gyyp) - B * Gzzr + e * dtp
        waver = m * dttr - B * (Gxxp + Gyyp) - Gzzr + e * dtr
        stencilp = solve(wavep, p(x, y, z, t+s), simplify=False)[0]
        stencilr = solve(waver, r(x, y, z, t+s), simplify=False)[0]
        return (stencilp, stencilr, (m, A, B, Th, Ph, s, h, e))

    def smart_sympy_replace(self, num_dim, time_order, res, funs, arrs, fw):
        a = Wild('a')
        b = Wild('b')
        c = Wild('c')
        d = Wild('d')
        e = Wild('e')
        f = Wild('f')
        q = Wild('q')
        x, y, z = symbols("x y z")
        h, s, t = symbols("h s t")
        width_t = int(time_order/2)
        assert(len(funs) == len(arrs))
        for fun, arr in zip(funs, arrs):
            if num_dim == 2:
                # Replace function notation with array notation
                res = res.replace(fun(a, b, c), arr[a, b, c])
                res = res.replace(arr[a*x+b, c*z+d, e*t+f], arr[e*t+f, a*x+b, c*z+d])
            else:  # num_dim == 3
                res = res.replace(fun(a, b, c, d), arr[a, b, c, d])
                res = res.replace(arr[x+b, y+q, z+d, t+f], arr[t+f, x+b, y+q, z+d])
        # Replace x+h in indices with x+1
        for dim_var in [x, y, z]:
            res = res.replace(dim_var+c*h, dim_var+c)
        # Replace t+s with t+1
        res = res.replace(t+c*s, t+c)
        if fw:
            res = res.subs({t: t-width_t})
        else:
            res = res.subs({t: t+width_t})
        return res

    def total_dim(self, ndim):
        if ndim == 2:
            return (t, x, z)
        else:
            return (t, x, y, z)

    def space_dim(self, ndim):
        if ndim == 2:
            return (x, z)
        else:
            return (x, y, z)


class ForwardOperator(TTIOperator):
    def __init__(self, m, src, damp, rec, u, v, a, b, th, ph, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)
        self.input_params = [m, src, damp, rec, u, v, a, b, th, ph]
        u.pad_time = False
        dt = src.dt
        h = src.h
        old_src = src
        self.output_params = []
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        stencilp, stencilr, subs = self._init_taylor(dim, time_order, spc_order)
        stencilp = self.smart_sympy_replace(dim, time_order, stencilp, [Function('p'), Function('r')], [u, v], fw=True)
        stencilr = self.smart_sympy_replace(dim, time_order, stencilr, [Function('p'), Function('r')], [u, v], fw=True)
        stencil_args = [m[space_dim], a[space_dim], b[space_dim], th[space_dim], ph[space_dim], dt, h, damp[space_dim]]
        first_stencil = Eq(u[total_dim], stencilp)
        second_stencil = Eq(v[total_dim], stencilr)
        self.stencils = [(first_stencil, stencil_args), (second_stencil, stencil_args)]
        src_list = old_src.add(m, u) + old_src.add(m, v)
        rec = rec.read(u, v)
        self.time_loop_stencils_post = src_list+rec
        super(ForwardOperator, self).__init__(subs, old_src.nt, m.shape, spc_border=spc_order/2, time_order=time_order, forward=True, dtype=m.dtype)


class AdjointOperator(TTIOperator):
    def __init__(self, m, rec, damp, srca, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)
        self.input_params = [m, rec, damp, srca]
        v = TimeData("v", m.shape, rec.nt, time_order=time_order, save=True, dtype=m.dtype)
        self.output_params = [v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'), v, fw=False)
        main_stencil = Eq(lhs, stencil)
        stencil_args = [m[space_dim], rec.dt, rec.h, damp[space_dim]]
        self.stencils = [(main_stencil, stencil_args)]
        rec_list = rec.add(m, v)
        src_list = srca.read(v)
        self.time_loop_stencils_post = rec_list + src_list
        super(AdjointOperator, self).__init__(subs, rec.nt, m.shape, spc_border=spc_order/2, time_order=time_order, forward=False, dtype=m.dtype)


class GradientOperator(TTIOperator):
    def __init__(self, u, m, rec, damp, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)
        self.input_params = [u, m, rec, damp]
        v = TimeData("v", m.shape, rec.nt, time_order=time_order, save=False, dtype=m.dtype)
        grad = DenseData("grad", m.shape, dtype=m.dtype)
        self.output_params = [grad, v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'), v, fw=False)
        stencil_args = [m[space_dim], rec.dt, rec.h, damp[space_dim]]
        main_stencil = Eq(lhs, lhs + stencil)
        gradient_update = Eq(grad[space_dim], grad[space_dim] - (v[total_dim] - 2 * v[tuple((t + 1,) + space_dim)] + v[tuple((t + 2,) + space_dim)]) * u[total_dim])
        reset_v = Eq(v[tuple((t + 2,) + space_dim)], 0)
        self.stencils = [(main_stencil, stencil_args), (gradient_update, []), (reset_v, [])]

        rec_list = rec.add(m, v)
        self.time_loop_stencils_pre = rec_list
        super(GradientOperator, self).__init__(subs, rec.nt, m.shape, spc_border=spc_order/2, time_order=time_order, forward=False, dtype=m.dtype)


class BornOperator(TTIOperator):
    def __init__(self, dm, m, src, damp, rec, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)
        self.input_params = [dm, m, src, damp, rec]
        u = TimeData("u", m.shape, src.nt, time_order=time_order, save=False, dtype=m.dtype)
        U = TimeData("U", m.shape, src.nt, time_order=time_order, save=False, dtype=m.dtype)
        self.output_params = [u, U]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        dt = src.dt
        h = src.h
        src_list = src.add(m, u)
        rec = rec.read(U)
        self.time_loop_stencils_pre = src_list
        self.time_loop_stencils_post = rec
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[0]
        first_stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'), u, fw=True)
        second_stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'), U, fw=True)
        first_stencil_args = [m[space_dim], dt, h, damp[space_dim]]
        first_update = Eq(u[total_dim], u[total_dim]+first_stencil)
        src2 = -(dt**-2)*(u[total_dim]-2*u[tuple((t - 1,) + space_dim)]+u[tuple((t - 2,) + space_dim)])*dm[space_dim]
        second_stencil_args = [m[space_dim], dt, h, damp[space_dim]]
        second_update = Eq(U[total_dim], second_stencil)
        insert_second_source = Eq(U[total_dim], U[total_dim]+(dt*dt)/m[space_dim]*src2)
        reset_u = Eq(u[tuple((t - 2,) + space_dim)], 0)
        self.stencils = [(first_update, first_stencil_args), (second_update, second_stencil_args), (insert_second_source, []), (reset_u, [])]
        super(BornOperator, self).__init__(subs, src.nt, m.shape, spc_border=spc_order/2, time_order=time_order, forward=True, dtype=m.dtype)

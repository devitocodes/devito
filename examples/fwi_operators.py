from devito.operators import *
from sympy import Eq
from devito.interfaces import TimeData, DenseData
from sympy import Function, symbols, as_finite_diff, Wild, IndexedBase
from sympy.abc import x, y, t, z
from sympy import solve


class FWIOperator(Operator):
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
        width_t = int(time_order/2)
        width_h = int(space_order/2)
        p = Function('p')
        s, h = symbols('s h')
        if dim == 2:
            m = IndexedBase("M")[x, z]
            e = IndexedBase("E")[x, z]
            solvep = p(x, z, t + width_t*s)
            solvepa = p(x, z, t - width_t*s)
        else:
            m = IndexedBase("M")[x, y, z]
            e = IndexedBase("E")[x, y, z]
            solvep = p(x, y, z, t + width_t*s)
            solvepa = p(x, y, z, t - width_t*s)

        # Indexes for finite differences
        indx = []
        indy = []
        indz = []
        indt = []
        for i in range(-width_h, width_h + 1):
            indx.append(x + i * h)
            indy.append(y + i * h)
            indz.append(z + i * h)

        for i in range(-width_t, width_t + 1):
            indt.append(t + i * s)
        # Time and space  discretization as a Taylor expansion.
        #
        # The time discretization is define as a second order ( $ O (dt^2)) $)
        # centered finite difference to get an explicit Euler scheme easy to
        # solve by steping in time.
        #
        # $ \frac{d^2 u(x,t)}{dt^2} \simeq \frac{u(x,t+dt) - 2 u(x,t) +
        # u(x,t-dt)}{dt^2} + O(dt^2) $
        #
        # And we define the space discretization also as a Taylor serie, with
        # oder chosen by the user. This can either be a direct expansion of the
        # second derivative bulding the laplacian, or a combination of first
        # oder space derivative. The second option can be a better choice in
        # case you would want to extand the method to more complex wave
        # equations involving first order derivatives in chain only.
        #
        # $ \frac{d^2 u(x,t)}{dt^2} \simeq \frac{1}{dx^2} \sum_k \alpha_k
        # (u(x+k dx,t)+u(x-k dx,t)) + O(dx^k) $

        # Finite differences
        if dim == 2:
            dtt = as_finite_diff(p(x, z, t).diff(t, t), indt)
            dxx = as_finite_diff(p(x, z, t).diff(x, x), indx)
            dzz = as_finite_diff(p(x, z, t).diff(z, z), indz)
            dt = as_finite_diff(p(x, z, t).diff(t), indt)
            lap = dxx + dzz
        else:
            dtt = as_finite_diff(p(x, y, z, t).diff(t, t), indt)
            dxx = as_finite_diff(p(x, y, z, t).diff(x, x), indx)
            dyy = as_finite_diff(p(x, y, z, t).diff(y, y), indy)
            dzz = as_finite_diff(p(x, y, z, t).diff(z, z), indz)
            dt = as_finite_diff(p(x, y, z, t).diff(t), indt)
            lap = dxx + dyy + dzz

        # Argument list
        arglamb = []
        arglamba = []
        if dim == 2:
            for i in range(-width_t, width_t):
                arglamb.append(p(x, z, indt[i + width_t]))
                arglamba.append(p(x, z, indt[i + width_t + 1]))

            for i in range(-width_h, width_h + 1):
                for j in range(-width_h, width_h + 1):
                    arglamb.append(p(indx[i + width_h], indz[j + width_h], t))
                    arglamba.append(p(indx[i + width_h], indz[j + width_h], t))
        else:
            for i in range(-width_t, width_t):
                arglamb.append(p(x, y, z, indt[i + width_t]))
                arglamba.append(p(x, y, z, indt[i + width_t + 1]))

            for i in range(-width_h, width_h+1):
                for j in range(-width_h, width_h+1):
                    for k in range(-width_h, width_h+1):
                        arglamb.append(p(indx[i + width_h], indy[i + width_h], indz[j + width_h], t))
                        arglamba.append(p(indx[i + width_h], indy[i + width_h], indz[j + width_h], t))

        arglamb.extend((m, s, h, e))
        arglamb = tuple(arglamb)
        arglamba.extend((m, s, h, e))
        arglamba = tuple(arglamba)

        wave_equation = m*dtt - lap + e*dt
        stencil = solve(wave_equation, solvep)[0]

        wave_equationA = m*dtt - lap - e*dt
        stencilA = solve(wave_equationA, solvepa)[0]
        return ((stencil, (m, s, h, e)), (stencilA, (m, s, h, e)))

    def smart_sympy_replace(self, num_dim, expr, fun, arr, fw):
        a = Wild('a')
        b = Wild('b')
        c = Wild('c')
        d = Wild('d')
        e = Wild('e')
        f = Wild('f')
        q = Wild('q')
        x, y, z = symbols("x y z")
        h, s, t = symbols("h s t")
        if num_dim == 2:
            # Replace function notation with array notation
            res = expr.replace(fun(a, b, c), arr[a, b, c])
            # Reorder indices so time comes first
            res = res.replace(arr[a*x+b, c*z+d, e*t+f], arr[e*t+f, a*x+b, c*z+d])
        if num_dim == 3:
            res = expr.replace(fun(a, b, c, d), arr[a, b, c, d])
            res = res.replace(arr[x+b, y+q, z+d, t+f], arr[t+f, x+b, y+q, z+d])
        # Replace x+h in indices with x+1
        for dim_var in [x, y, z]:
            res = res.replace(dim_var+c*h, dim_var+c)
        # Replace t+s with t+1
        res = res.replace(t+c*s, t+c)
        if fw:
            res = res.subs({t: t-1})
        else:
            res = res.subs({t: t+1})
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


class ForwardOperator(FWIOperator):
    def __init__(self, m, src, damp, rec, u, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)
        self.input_params = [m, src, damp, rec, u]
        u.pad_time = True
        self.output_params = []
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[0]
        stencil = self.smart_sympy_replace(dim, stencil, Function('p'), u, fw=True)
        stencil_args = [m[space_dim], src.dt, src.h, damp[space_dim]]
        main_stencil = Eq(u[total_dim], stencil)
        self.stencils = [(main_stencil, stencil_args)]
        src_list = src.add(m, u)
        rec = rec.read(u)
        self.time_loop_stencils_post = src_list+rec
        super(ForwardOperator, self).__init__(subs, src.nt, m.shape, spc_border=spc_order/2, time_order=time_order, forward=True, dtype=m.dtype)


class AdjointOperator(FWIOperator):
    def __init__(self, m, rec, damp, srca, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)
        self.input_params = [m, rec, damp, srca]
        v = TimeData("v", m.shape, rec.nt, time_order=2, save=True, dtype=m.dtype)
        self.output_params = [v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, stencil, Function('p'), v, fw=False)
        main_stencil = Eq(lhs, stencil)
        stencil_args = [m[space_dim], rec.dt, rec.h, damp[space_dim]]
        self.stencils = [(main_stencil, stencil_args)]
        rec_list = rec.add(m, v)
        src_list = srca.read(v)
        self.time_loop_stencils_post = rec_list + src_list
        super(AdjointOperator, self).__init__(subs, rec.nt, m.shape, spc_border=spc_order/2, time_order=time_order, forward=False, dtype=m.dtype)


class GradientOperator(FWIOperator):
    def __init__(self, u, m, rec, damp, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)
        self.input_params = [u, m, rec, damp]
        v = TimeData("v", m.shape, rec.nt, time_order=2, save=False, dtype=m.dtype)
        grad = DenseData("grad", m.shape, dtype=m.dtype)
        self.output_params = [grad, v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, stencil, Function('p'), v, fw=False)
        stencil_args = [m[space_dim], rec.dt, rec.h, damp[space_dim]]
        main_stencil = Eq(lhs, lhs + stencil)
        gradient_update = Eq(grad[space_dim], grad[space_dim] - (v[total_dim] - 2 * v[tuple((t + 1,) + space_dim)] + v[tuple((t + 2,) + space_dim)]) * u[total_dim])
        reset_v = Eq(v[tuple((t + 2,) + space_dim)], 0)
        self.stencils = [(main_stencil, stencil_args), (gradient_update, []), (reset_v, [])]

        rec_list = rec.add(m, v)
        self.time_loop_stencils_pre = rec_list
        super(GradientOperator, self).__init__(subs, rec.nt, m.shape, spc_border=spc_order/2, time_order=time_order, forward=False, dtype=m.dtype)


class BornOperator(FWIOperator):
    def __init__(self, dm, m, src, damp, rec, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)
        self.input_params = [dm, m, src, damp, rec]
        u = TimeData("u", m.shape, src.nt, time_order=2, save=False, dtype=m.dtype)
        U = TimeData("U", m.shape, src.nt, time_order=2, save=False, dtype=m.dtype)
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
        first_stencil = self.smart_sympy_replace(dim, stencil, Function('p'), u, fw=True)
        second_stencil = self.smart_sympy_replace(dim, stencil, Function('p'), U, fw=True)
        first_stencil_args = [m[space_dim], dt, h, damp[space_dim]]
        first_update = Eq(u[total_dim], u[total_dim]+first_stencil)
        src2 = -(dt**-2)*(u[total_dim]-2*u[tuple((t - 1,) + space_dim)]+u[tuple((t - 2,) + space_dim)])*dm[space_dim]
        second_stencil_args = [m[space_dim], dt, h, damp[space_dim]]
        second_update = Eq(U[total_dim], second_stencil)
        insert_second_source = Eq(U[total_dim], U[total_dim]+(dt*dt)/m[space_dim]*src2)
        reset_u = Eq(u[tuple((t - 2,) + space_dim)], 0)
        self.stencils = [(first_update, first_stencil_args), (second_update, second_stencil_args), (insert_second_source, []), (reset_u, [])]
        super(BornOperator, self).__init__(subs, src.nt, m.shape, spc_border=spc_order/2, time_order=time_order, forward=True, dtype=m.dtype)

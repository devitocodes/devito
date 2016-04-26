# coding: utf-8
from __future__ import print_function
from sympy import Function, symbols, init_printing, as_finite_diff
from sympy import solve, IndexedBase, Eq
from sympy.abc import x, y, t, M, Q, D, E
import numpy as np
from generator import Generator
from function_descriptor import FunctionDescriptor
from propagator import Propagator

init_printing()


class AcousticWave2D_cg:

    def __init__(self, model, data, nbpml=10):
        self.model = model
        self.data = data
        self.dtype = np.float64
        self.dt = model.get_critical_dt()
        self.h = model.get_spacing()[0]
        self.nbpml = nbpml
        self.src_grid = None

    def prepare(self, nt):
        self._init_taylor(nt)
        self._forwardStencil, self._adjointStencil = self.g.get_wrapped_functions()

    def _init_taylor(self, nt):
        # The acoustic wave equation for the square slowness m and a source q
        # is given in 3D by :
        #
        # \begin{cases} &m \frac{d^2 u(x,t)}{dt^2} - \nabla^2 u(x,t) =q  \\
        # &u(.,0) = 0 \\ &\frac{d u(x,t)}{dt}|_{t=0} = 0 \end{cases}
        #
        # with the zero initial conditons to guarantee unicity of the solution

        p = Function('p')
        s, h = symbols('s h')
        m = M(x, y)
        q = Q(x, y, t)
        d = D(x, y, t)
        e = E(x, y)
        nx, ny = self.model.get_dimensions()
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

        dtt = as_finite_diff(p(x, y, t).diff(t, t), [t-s, t, t+s])
        dt = as_finite_diff(p(x, y, t).diff(t), [t-s, t+s])

        # Spacial finite differences can easily be extended to higher order by
        # increasing the list of sampling point in the next expression.  Be
        # sure to keep this stencil symmetric and everything else in the
        # notebook will follow.
        dxx = as_finite_diff(p(x, y, t).diff(x, x), [x-h, x, x+h])
        dyy = as_finite_diff(p(x, y, t).diff(y, y), [y-h, y, y+h])

        # Solve forward in time
        #
        # The wave equation with absorbing boundary conditions writes
        #
        # $ \eta \frac{d u(x,t)}{dt} + \frac{d^2 u(x,t)}{dt^2} - \nabla^2
        # u(x,t) =q  $
        #
        # and the adjont wave equation
        #
        # $ -\eta \frac{d u(x,t)}{dt} + \frac{d^2 u(x,t)}{dt^2} - \nabla^2
        # u(x,t) =q  $
        #
        # where $ \eta$  is a damping factor equal to zero inside the physical
        # domain and decreasing inside the absorbing layer from the pysical
        # domain to the border

        # Forward wave equation
        wave_equation = m * dtt - (dxx + dyy) - q + e * dt
        stencil = solve(wave_equation, p(x, y, t+s))[0]
        self.nt = nt

        kernel = self._prepare_forward_kernel((p(x, y, t-s),
                                               p(x-h, y, t),
                                               p(x, y, t),
                                               p(x+h, y, t),
                                               p(x, y-h, t),
                                               p(x, y+h, t), q, m, s, h, e),
                                              stencil, nt)
        fd = FunctionDescriptor("Forward", kernel)
        fd.add_matrix_param("u", (nt, nx, ny), self.dtype)
        fd.add_matrix_param("rec", (nt, ny - 2), self.dtype)
        fd.add_matrix_param("m", (nx, ny), self.dtype)
        fd.add_matrix_param("src_grid", (nx, ny), self.dtype)
        fd.add_matrix_param("src_time", (nt,), self.dtype)
        fd.add_matrix_param("dampx", (nx,), self.dtype)
        fd.add_matrix_param("dampy", (ny,), self.dtype)

        fd.add_value_param("xrec", "int")
        fds = [fd]

        # Precompute dampening
        self.dampx = np.array([self.damp(i, nx) for i in range(nx)], dtype=self.dtype, order='C')
        self.dampy = np.array([self.damp(i, ny) for i in range(ny)], dtype=self.dtype, order='C')

        # Rewriting the discret PDE as part of an Inversion. Accuracy and
        # rigourousness of the dicretization
        #
        # The above axpression are good for modelling. However, if you want to
        # include a wave equation solver into an Inversion workflow, a more
        # rigourous study of the discretization must be done. We can rewrite a
        # single time step as follows
        #
        # $ A_3  u(x,t+dt)  = A_1 u(x,t) + A_2 u(x,t-dt) +q(x,t)$
        #
        # where $ A_1,A_2,A_3 $ are square, invertible matrices, and symetric
        # without any boundary conditions. In more details we have :
        #
        # \begin{align} & A_1 = \frac{2}{dt^2 m} + \Delta \\ & A_2 =
        # \frac{-1}{dt^2 m} \\ & A_3 = \frac{1}{dt^2 m} \end{align}
        #
        # We can the write the action of the adjoint wave equation operator.
        # The adjoint wave equation is defined by \begin{cases} &m \frac{d^2
        # v(x,t)}{dt^2} - \nabla^2 v(x,t) = \delta d  \\ &v(.,T) = 0 \\
        # &\frac{d v(x,t)}{dt}|_{t=T} = 0 \end{cases}
        #
        # but by choosing to discretize first we will not discretize this
        # equation. Instead we will take the adjoint of the forward wave
        # equation operator and by testing that the operator is the true
        # adjoint, we will guaranty solving the adjoint wave equation. We have
        # the the single time step for the adjoint wavefield going backward in
        # time in order to keep an explicit Euler scheme
        #
        # $  A_2^T v(x,t-dt)  = A_1^T v(x,t) + A_3^T v(x,t+dt) + \delta d(x,t)$
        #
        # and as $A_2$ and $A_3$ are diagonal matrices  there is no issue in
        # inverting it. We can also see that choosing a asymetric stencil for
        # the spacial derivative may lead to erro has the Laplacian would stop
        # to be self-adjoint, and the actual adjoint finite difference scheme
        # should be implemented.

        # Adjoint wave equation
        wave_equationA = m * dtt - (dxx + dyy) - D(x, y, t) - e * dt
        stencilA = solve(wave_equationA, p(x, y, t-s))[0]
        kernelA = self._prepare_adjoint_kernel((p(x, y, t+s),
                                                p(x-h, y, t),
                                                p(x, y, t),
                                                p(x+h, y, t),
                                                p(x, y-h, t),
                                                p(x, y+h, t), d, m, s, h, e),
                                               stencilA, nt)

        fd = FunctionDescriptor("Adjoint", kernelA)
        fd.add_matrix_param("v", (nt, nx, ny), self.dtype)
        fd.add_matrix_param("rec", (nt, ny - 2), self.dtype)
        fd.add_matrix_param("m", (nx, ny), self.dtype)
        fd.add_matrix_param("src_grid", (nx, ny), self.dtype)
        fd.add_matrix_param("srca", (nt,), self.dtype)
        fd.add_matrix_param("dampx", (nx,), self.dtype)
        fd.add_matrix_param("dampy", (ny,), self.dtype)
        fd.add_value_param("xrec", "int")
        fd.add_local_variable("resid", self.dtype)

        fds.append(fd)
        self.g = Generator(fds, dtype=self.dtype)

    def Forward(self):
        nx, ny = self.model.get_dimensions()
        m = self.model.vp**(-2)
        nt = self.nt
        u = np.zeros((nt, nx, ny), dtype=self.dtype)
        rec = np.zeros((nt, ny - 2), dtype=self.dtype)
        src_grid = self.source_interpolate()
        self._forwardStencil(u, rec, m, src_grid, self.data.get_source(),
                             self.dampx, self.dampy, int(self.data.receiver_coords[0, 2]))
        return rec, u

    def Adjoint(self, nt, rec):
        nx, ny = self.model.get_dimensions()
        m = self.model.vp**(-2)
        v = np.zeros((nt, nx, ny))
        srca = np.zeros((nt))
        src_grid = self.source_interpolate()
        self._adjointStencil(v, rec, m, src_grid, srca,
                             self.dampx, self.dampy, int(self.data.receiver_coords[0, 2]))
        return srca, v

    def Gradient(self, nt, rec, u):
        nx, ny = self.model.get_dimensions()
        dt = self.dt
        h = self.h
        m = self.model.get_vp()
        v1 = np.zeros((nx, ny))
        v2 = np.zeros((nx, ny))
        v3 = np.zeros((nx, ny))
        grad = np.zeros((nx, ny))
        for ti in range(nt-1, -1, -1):
            for a in range(1, nx-1):
                for b in range(1, ny-1):
                    if a == self.data.xrec:
                        resid = rec[ti, b-1]
                    else:
                        resid = 0
                    damp = self.dampx(a) + self.dampy(b)
                    v3[a, b] = self.tsA(v1[a, b],
                                        v2[a - 1, b],
                                        v2[a, b],
                                        v2[a + 1, b],
                                        v2[a, b - 1],
                                        v2[a, b + 1],
                                        resid, m[a, b],
                                        dt, h, damp)
                    grad[a, b] = grad[a, b] - \
                        (v3[a, b] - 2 * v2[a, b] + v1[a, b]) * (u[ti, a, b])
                    v1, v2, v3 = v2, v3, v1
        return dt*dt*grad

    def Born(self, nt, dm):
        xmin, ymin = self.model.get_origin()
        nx, ny = self.model.get_dimensions()
        dt = self.dt
        h = self.h
        m = self.model.get_vp()
        u1 = np.zeros((nx, ny))
        U1 = np.zeros((nx, ny))
        u2 = np.zeros((nx, ny))
        U2 = np.zeros((nx, ny))
        u3 = np.zeros((nx, ny))
        U3 = np.zeros((nx, ny))
        rec = np.zeros((nt, ny-2))
        src2 = 0
        for ti in range(0, nt):
            for a in range(1, nx-1):
                for b in range(1, ny-1):
                    damp = self.dampx(a)+self.dampy(b)
                    if self.source_interpolate(xmin+a*h, ymin+h*b):
                        src = self.data.get_source(ti)
                    else:
                        src = 0
                    u3[a, b] = self.ts(u1[a, b],
                                       u2[a - 1, b],
                                       u2[a, b],
                                       u2[a + 1, b],
                                       u2[a, b - 1],
                                       u2[a, b + 1],
                                       src, m[a, b], dt, h, damp)
                    src2 = -dt*dt*(u3[a, b]-2*u2[a, b]+u1[a, b])*dm[a, b]
                    U3[a, b] = self.ts(U1[a, b],
                                       U2[a - 1, b],
                                       U2[a, b],
                                       U2[a + 1, b],
                                       U2[a, b - 1],
                                       U2[a, b + 1],
                                       src2, m[a, b], dt, h, damp)
                    if a == self.data.xrec:
                        rec[ti, b-1] = U3[a, b]
            u1, u2, u3 = u2, u3, u1
            U1, U2, U3 = U2, U3, U1
        return rec

    def _prepare_forward_kernel(self, subs, stencil, nt):
        u = IndexedBase("u")
        M = IndexedBase("m")
        nx, ny = self.model.get_dimensions()
        src_time = IndexedBase("src_time")
        src_grid = IndexedBase("src_grid")
        dampx = IndexedBase("dampx")
        dampy = IndexedBase("dampy")
        rec = IndexedBase("rec")
        xrec = symbols("xrec")
        propagator = Propagator(nt, (nx, ny), 1)
        stencil1_args = [0, 0, 0, 0, 0, 0, (src_grid[x, y]*src_time[t]),
                         M[x, y], self.dt, self.h, dampx[x] + dampy[y]]
        stencil2_args = [0, u[t - 1, x - 1, y],
                         u[t - 1, x, y],
                         u[t - 1, x + 1, y],
                         u[t - 1, x, y - 1],
                         u[t - 1, x, y + 1],
                         src_grid[x, y]*src_time[t], M[x, y],
                         self.dt, self.h, dampx[x] + dampy[y]]
        stencil3_args = [u[t - 2, x, y],
                         u[t - 1, x - 1, y],
                         u[t - 1, x, y],
                         u[t - 1, x + 1, y],
                         u[t - 1, x, y - 1],
                         u[t - 1, x, y + 1],
                         (src_grid[x, y]*src_time[t]), M[x, y],
                         self.dt, self.h, dampx[x] + dampy[y]]
        lhs = u[t, x, y]
        # Sympy representation of if condition: x == xrec
        record_condition = Eq(x, xrec)
        # Sympy representation of assignment: rec[t, y-1] = u[t, x, y]
        record_true = Eq(rec[t, y-1], lhs)
        propagator.add_loop_step(record_condition, record_true)
        forward_loop = propagator.prepare(subs, stencil, stencil1_args,
                                          stencil2_args, stencil3_args, lhs)
        return forward_loop

    def _prepare_adjoint_kernel(self, subs, stencil, nt):
        nx, ny = self.model.get_dimensions()
        v = IndexedBase("v")
        M = IndexedBase("m")
        src_grid = IndexedBase("src_grid")
        srca = IndexedBase("srca")
        dampx = IndexedBase("dampx")
        dampy = IndexedBase("dampy")
        rec = IndexedBase("rec")
        xrec = symbols("xrec")
        resid = symbols("resid")
        dt = self.dt
        h = self.h
        use_receiver = Eq(resid, rec[t, y-1])
        dont_use_receiver = Eq(resid, 0)
        receiver_cond = Eq(x, xrec)
        propagator = Propagator(nt, (nx, ny), 1, False)
        propagator.add_loop_step(receiver_cond, use_receiver, dont_use_receiver, True)
        lhs = v[t, x, y]
        src_cond = Eq(src_grid[x, y], 1)
        src_cap = Eq(srca[t], lhs)
        propagator.add_loop_step(src_cond, src_cap)
        stencil1_args = [0, 0, 0, 0, 0, 0, resid, M[x, y], dt, h, dampx[x]+dampy[y]]
        stencil2_args = [0, v[t+1, x-1, y], v[t+1, x, y], v[t+1, x+1, y], v[t+1, x, y-1], v[t+1, x, y+1], resid, M[x, y], dt, h, dampx[x]+dampy[y]]
        stencil3_args = [v[t + 2, x, y], v[t + 1, x - 1, y], v[t + 1, x, y], v[t + 1, x + 1, y], v[t + 1, x, y - 1], v[t + 1, x, y + 1], resid, M[x, y], dt, h, dampx[x]+dampy[y]]
        backward_loop = propagator.prepare(subs, stencil, stencil1_args,
                                           stencil2_args, stencil3_args, lhs)
        return backward_loop

    def source_interpolate(self):
        if self.src_grid is not None:
            return self.src_grid
        xmin, ymin = self.model.get_origin()
        nx, ny = self.model.get_dimensions()
        h = self.h
        src_grid = np.zeros((nx, ny), dtype=self.dtype)
        for a in range(nx):
            for b in range(ny):
                sa, sb = xmin + a * h, ymin + h * b
                if (abs(sa - self.data.source_coords[0]) < self.h / 2 and
                        abs(sb - self.data.source_coords[1]) < self.h / 2):
                    src_grid[a, b] = 1
                else:
                    src_grid[a, b] = 0
        self.src_grid = src_grid
        return src_grid

    def damp(self, x, nx):
        nbpml = self.nbpml
        h = self.h

        dampcoeff = 1.5 * np.log(1.0 / 0.001) / (5.0 * h)
        if x < nbpml:
            return dampcoeff * ((nbpml - x) / nbpml)**2
        elif x > nx - nbpml - 1:
            return dampcoeff * ((x - nx + nbpml) / nbpml)**2
        else:
            return 0.0

    def compute_gradient(self, model, shot_id):
        raise NotImplementedError("compute_gradient")

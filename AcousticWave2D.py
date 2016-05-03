from __future__ import print_function

from sympy import Function, symbols, init_printing, as_finite_diff, solve, \
                  lambdify, Matrix
from sympy.abc import x, y, t, M, E

import sympy
import numpy as np
from numpy import linalg
from math import floor

init_printing()


class AcousticWave2D:
    def __init__(self, model, data, source=None, nbpml=10):
        self.model = model
        self.data = data
        self.nbpml = nbpml
        self.dt = model.get_critical_dt()
        self.h = model.get_spacing()
        self.nt = self.data.nt
        pad = ((nbpml, nbpml), (nbpml, nbpml))
        self.model.vp = np.pad(self.model.vp, pad, 'edge')
        self.nrec = self.data.get_nrec()
        self._init_taylor()
        
        if source is not None:
           self.source = source.read()
           self.source.reinterpolate(self.dt)
           source_time = self.source.traces[0,:]
           if len(source_time) < self.data.nsamples:
               source_time = np.append(source_time, [0.0])
           self.data.set_source(source_time, self.dt, self.data.source_coords)
        # Set up interpolation from grid to receiver position.
        x1, z1, x2, z2, d = symbols('x1, z1, x2, z2, d')
        A = Matrix([[1, x1, z1, x1*z1],
                    [1, x1, z2, x1*z2],
                    [1, x2, z1, x2*z1],
                    [1, x2, z2, x2*z2]])

        # Map to reference cell
        reference_cell = [(x1, 0),
                          (z1, 0),
                          (x2, self.h),
                          (z2, self.h)]
        A = A.subs(reference_cell)
        
        # Form expression for interpolant weights on reference cell.
        self.rs = sympy.symbols('rx, rz')
        rx, rz = self.rs
        p = sympy.Matrix([[1],
                          [rx],
                          [rz],
                          [rx*rz]])

        b11, b12, b21, b22 = A.inv().T.dot(p)
        self.bs = b11, b12, b21, b22
        self.nt = self.data.nt
    
    def prepare(self):
        pass

    def grid2point(self, u, x, z):
        rx, rz = self.rs
        b11, b12, b21, b22 = self.bs
        i = int(x/self.h)
        j = int(z/self.h)

        x = x - i*self.h
        z = z - j*self.h

        return (b11.subs(((rx, x), (rz, z))) * u[i, j] +
                b12.subs(((rx, x), (rz, z))) * u[i, j+1] +
                b21.subs(((rx, x), (rz, z))) * u[i+1, j] +
                b22.subs(((rx, x), (rz, z))) * u[i+1, j+1])

    # Interpolate source onto grid.
    def point2grid(self, x, z):
        rx, rz = self.rs
        b11, b12, b21, b22 = self.bs
        # In: s - Magnitude of the source
        #     x, z - Position of the source
        # Returns: (i, k) - Grid coordinate at top left of grid cell.
        #          (s11, s12, s21, s22) - source values at coordinates
        #          (i, k), (i, k+1), (i+1, k), (i+1, k+1)
        i = int(x/self.h)
        k = int(z/self.h)

        x = x - i*self.h
        z = z - k*self.h

        s11 = b11.subs(((rx, x), (rz, z))).evalf()
        s12 = b12.subs(((rx, x), (rz, z))).evalf()
        s21 = b21.subs(((rx, x), (rz, z))).evalf()
        s22 = b22.subs(((rx, x), (rz, z))).evalf()
        return (i, k), (s11, s12, s21, s22)

    def _init_taylor(self, h=None, s=None):
        # The acoustic wave equation for the square slowness m and a source q
        # is given in 3D by :
        #
        # \begin{cases} &m \frac{d^2 u(x,t)}{dt^2} - \nabla^2 u(x,t) =q  \\
        # &u(.,0) = 0 \\ &\frac{d u(x,t)}{dt}|_{t=0} = 0 \end{cases}
        #
        # with the zero initial conditons to guarantee unicity of the solution

        p = Function('p')
        if s is None:
            s = symbols('s')
        if h is None:
            h = symbols('h')
        m = M(x, y)
        e = E(x, y)

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
        wave_equation = m * dtt - (dxx + dyy) + e * dt
        stencil = solve(wave_equation, p(x, y, t+s))[0]

        self.ts = lambdify((p(x, y, t-s),
                           p(x-h, y, t),
                           p(x, y, t),
                           p(x+h, y, t),
                           p(x, y-h, t),
                           p(x, y+h, t), m, s, h, e),
                           stencil, "numpy")

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
        wave_equationA = m * dtt - (dxx + dyy) - e * dt
        stencilA = solve(wave_equationA, p(x, y, t-s))[0]
        self.tsA = lambdify((p(x, y, t+s),
                            p(x-h, y, t),
                            p(x, y, t),
                            p(x+h, y, t),
                            p(x, y-h, t),
                            p(x, y+h, t), m, s, h, e),
                            stencilA, "numpy")

    def dampx(self, x):
        nbpml = self.nbpml
        h = self.h
        nx, ny = self.model.get_shape()
        dampcoeff = 1.5 * np.log(1.0 / 0.001) / (5.0 * h)
        if x < nbpml:
            return dampcoeff * ((nbpml - x) / nbpml)**2
        elif x > nx - nbpml - 1:
            return dampcoeff * ((x - nx + nbpml) / nbpml)**2
        else:
            return 0.0

    def dampy(self, y):
        nbpml = self.nbpml
        h = self.h
        nx, ny = self.model.get_shape()

        dampcoeff = 1.5 * np.log(1.0 / 0.001) / (5.0 * h)
        if y < nbpml:
            return dampcoeff*((nbpml-y)/nbpml)**2
        elif y > ny - nbpml - 1:
            return dampcoeff * ((y - ny + nbpml) / nbpml)**2
        else:
            return 0.0

    # Interpolate source onto grid...
    def add_source(self, s, m, dt, u):
        src_add = self.point2grid(self.data.source_coords[0],
                                  self.data.source_coords[2])
        (i, k) = src_add[0]
        u[i, k] = u[i, k]+s*dt*dt/m[i, k]*src_add[1][0]
        u[i, k+1] = u[i, k+1]+s*dt*dt/m[i, k]*src_add[1][1]
        u[i+1, k] = u[i+1, k]+s*dt*dt/m[i, k]*src_add[1][2]
        u[i+1, k+1] = u[i+1, k+1]+s*dt*dt/m[i, k]*src_add[1][3]

    def read_source(self, u):
        return self.grid2point(u, self.data.source_coords[0],
                               self.data.source_coords[2])

    def add_rec(self, rec, m, dt, u):
        ntraces, nsamples = self.data.traces.shape
        for j in range(ntraces):
            rec_add = self.point2grid(self.data.receiver_coords[j, 0],
                                      self.data.receiver_coords[j, 2])
            (i, k) = rec_add[0]
            u[i, k] = u[i, k]+rec[j]*dt*dt/m[i, k]*rec_add[1][0]
            u[i, k+1] = u[i, k+1]+rec[j]*dt*dt/m[i, k]*rec_add[1][1]
            u[i+1, k] = u[i+1, k]+rec[j]*dt*dt/m[i, k]*rec_add[1][2]
            u[i+1, k+1] = u[i+1, k+1]+rec[j]*dt*dt/m[i, k]*rec_add[1][3]

    def read_rec(self, rec, u):
        ntraces, nsamples = self.data.traces.shape
        for i in range(ntraces):
            rec[i] = self.grid2point(u, self.data.receiver_coords[i, 0],
                                     self.data.receiver_coords[i, 2])

    def Forward(self):
        nt = self.nt
        nx, ny = self.model.get_shape()
        m = self.model.vp**(-2)
        dt = self.dt
        h = self.h

        u = np.zeros((nt, nx, ny))
        rec = np.zeros((nt, self.nrec))
        for ti in range(0, nt):
            for a in range(1, nx-1):
                for b in range(1, ny-1):
                    damp = self.dampx(a) + self.dampy(b)
                    if ti == 0:
                        u[ti, a, b] = self.ts(0, 0, 0, 0, 0, 0,
                                              m[a, b], dt, h, damp)
                    elif ti == 1:
                        u[ti, a, b] = self.ts(0, u[ti - 1, a - 1, b],
                                              u[ti - 1, a, b],
                                              u[ti - 1, a + 1, b],
                                              u[ti - 1, a, b - 1],
                                              u[ti - 1, a, b + 1],
                                              m[a, b], dt, h, damp)
                    else:
                        u[ti, a, b] = self.ts(u[ti - 2, a, b],
                                              u[ti - 1, a - 1, b],
                                              u[ti - 1, a, b],
                                              u[ti - 1, a + 1, b],
                                              u[ti - 1, a, b - 1],
                                              u[ti - 1, a, b + 1],
                                              m[a, b], dt, h, damp)
            self.add_source(self.data.get_source(ti), m, dt, u[ti, :, :])
            self.read_rec(rec[ti, :], u[ti, :, :])
        return rec, u

    def Adjoint(self, rec):
        nt = self.nt
        nx, ny = self.model.get_shape()
        dt = self.dt
        h = self.h
        m = self.model.vp**(-2)
        v = np.zeros((nt, nx, ny))
        srca = np.zeros((nt))
        for ti in range(nt - 1, -1, -1):
            for a in range(1, nx-1):
                for b in range(1, ny-1):
                    damp = self.dampx(a) + self.dampy(b)
                    if ti == nt-1:
                        v[ti, a, b] = self.tsA(0, 0, 0, 0, 0, 0,
                                               m[a, b], dt, h, damp)
                    elif ti == nt-2:
                        v[ti, a, b] = self.tsA(0,
                                               v[ti+1, a-1, b],
                                               v[ti+1, a, b],
                                               v[ti+1, a+1, b],
                                               v[ti+1, a, b-1],
                                               v[ti+1, a, b+1],
                                               m[a, b],
                                               dt, h, damp)
                    else:
                        v[ti, a, b] = self.tsA(v[ti + 2, a, b],
                                               v[ti + 1, a - 1, b],
                                               v[ti + 1, a, b],
                                               v[ti + 1, a + 1, b],
                                               v[ti + 1, a, b - 1],
                                               v[ti + 1, a, b + 1],
                                               m[a, b], dt, h, damp)
            self.add_rec(rec[ti, :], m, dt, v[ti, :, :])
            srca[ti] = self.read_source(v[ti, :, :])
        return srca, v

    def Gradient(self, rec, u):
        nx, ny = self.model.get_shape()
        dt = self.dt
        h = self.h
        nt = self.nt
        m = self.model.vp**(-2)
        v1 = np.zeros((nx, ny))
        v2 = np.zeros((nx, ny))
        v3 = np.zeros((nx, ny))
        grad = np.zeros((nx, ny))

        for ti in range(nt-1, -1, -1):
            v3[:, :] = 0.0
            self.add_rec(rec[ti, :], m, dt, v3[:, :])
            for a in range(1, nx-1):
                for b in range(1, ny-1):
                    damp = self.dampx(a) + self.dampy(b)
                    v3[a, b] = v3[a, b] + self.tsA(v1[a, b],
                                                   v2[a - 1, b],
                                                   v2[a, b],
                                                   v2[a + 1, b],
                                                   v2[a, b - 1],
                                                   v2[a, b + 1],
                                                   m[a, b], dt, h, damp)
                    grad[a, b] = grad[a, b] - \
                        (v3[a, b] - 2 * v2[a, b] + v1[a, b]) * (u[ti, a, b])
            v1, v2, v3 = v2, v3, v1
        return (dt**-2)*grad

    def Born(self, dm):
        nt = self.nt
        nx, ny = self.model.get_shape()
        dt = self.dt
        h = self.h
        m = self.model.vp**(-2)
        u1 = np.zeros((nx, ny))
        U1 = np.zeros((nx, ny))
        u2 = np.zeros((nx, ny))
        U2 = np.zeros((nx, ny))
        u3 = np.zeros((nx, ny))
        U3 = np.zeros((nx, ny))
        rec = np.zeros((nt, self.nrec))
        src2 = 0
        for ti in range(0, nt):
            for a in range(1, nx-1):
                for b in range(1, ny-1):
                    damp = self.dampx(a) + self.dampy(b)
                    u3[a, b] = self.ts(u1[a, b],
                                       u2[a - 1, b],
                                       u2[a, b],
                                       u2[a + 1, b],
                                       u2[a, b - 1],
                                       u2[a, b + 1],
                                       m[a, b], dt, h, damp)
            self.add_source(self.data.get_source(ti), m, dt, u3)
            for a in range(1, nx-1):
                for b in range(1, ny-1):
                    damp = self.dampx(a) + self.dampy(b)
                    src2 = -(dt**-2)*(u3[a, b]-2*u2[a, b]+u1[a, b])*dm[a, b]
                    U3[a, b] = self.ts(U1[a, b],
                                       U2[a - 1, b],
                                       U2[a, b],
                                       U2[a + 1, b],
                                       U2[a, b - 1],
                                       U2[a, b + 1],
                                       m[a, b], dt, h, damp)
                    U3[a, b] = U3[a, b] + dt*dt/m[a, b] * src2
            self.read_rec(rec[ti, :], U3)
            u1, u2, u3 = u2, u3, u1
            U1, U2, U3 = U2, U3, U1
        return rec

    def run(self):
        nt = self.data.nsamples
        print('Starting forward')
        rec, u = self.Forward(nt)
        res = rec - np.transpose(self.data.traces)
        f = 0.5*linalg.norm(res)**2
        print('Residual is ', f, 'starting gradient')
        g = self.Gradient(nt, res, u)
        return f, g
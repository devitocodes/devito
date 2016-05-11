# coding: utf-8
from __future__ import print_function
from sympy import Function, symbols, init_printing, as_finite_diff
from sympy import solve, Matrix
from sympy.abc import x, y, t, M, E
import numpy as np
from operators import *
from devito.interfaces import DenseData, PointData

init_printing()


class AcousticWave2D_cg:

    def __init__(self, model, data, dm_initializer, source=None, nbpml=40):
        self.model = model
        self.data = data
        self.dtype = np.float64
        self.dt = model.get_critical_dt()
        self.h = model.get_spacing()
        self.nbpml = nbpml
        pad = ((nbpml, nbpml), (nbpml, nbpml))
        self.model.vp = np.pad(self.model.vp, pad, 'edge')
        self.data.reinterpolate(self.dt)
        self.nrec, self.nt = self.data.traces.shape
        self.model.set_origin(nbpml)
        self.dm_initializer = dm_initializer
        if source is not None:
            self.source = source.read()
            self.source.reinterpolate(self.dt)
            source_time = self.source.traces[0, :]
            if len(source_time) < self.data.nsamples:
                source_time = np.append(source_time, [0.0])
            self.data.set_source(source_time, self.dt, self.data.source_coords)

        self._init_taylor(self.nt)

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
        e = E(x, y)
        nx, ny = self.model.get_shape()
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

        def damp_boundary(damp):
            h = self.h
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40 * h)
            nx, ny = self.model.get_shape()
            nbpml = self.nbpml
            for i in range(nbpml):
                pos = np.abs((nbpml-i)/nbpml)
                val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
                damp[i, :] += val
                damp[-i, :] += val
                damp[:, i] += val
                damp[:, -i] += val

        wave_equation = m * dtt - (dxx + dyy) + e * dt
        stencil = solve(wave_equation, p(x, y, t+s))[0]
        self.nt = nt
        m_sub = DenseData("m", self.model.vp.shape, self.dtype)
        m_sub.initializer = lambda ref: np.copyto(ref, self.model.vp**(-2))
        self.m = m_sub
        damp = DenseData("damp", self.model.vp.shape, self.dtype)
        damp.initializer = damp_boundary
        self.damp = damp
        src = SourceLike("src", 1, self.nt, self.dt, self.h, np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :], self.dtype)
        self.src = src
        rec = SourceLike("rec", self.nrec, self.nt, self.dt, self.h, self.data.receiver_coords, self.dtype)
        src.initializer = lambda ref: np.copyto(ref, self.data.get_source()[:, np.newaxis])
        self.rec = rec
        u = TimeData("u", m_sub.shape, src.nt, time_order=2, save=True, dtype=self.dtype)
        self.u = u
        self._forward_stencil = ForwardOperator((p(x, y, t-s),
                                                 p(x-h, y, t),
                                                 p(x, y, t),
                                                 p(x+h, y, t),
                                                 p(x, y-h, t),
                                                 p(x, y+h, t), m, s, h, e),
                                                stencil, m_sub, src, damp, rec, u)

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
        srca = SourceLike("srca", 1, self.nt, self.dt, self.h, np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :], self.dtype)
        self.srca = srca
        wave_equationA = m * dtt - (dxx + dyy) - e * dt
        stencilA = solve(wave_equationA, p(x, y, t-s))[0]
        self._adjoint_stencil = AdjointOperator((p(x, y, t+s),
                                                 p(x-h, y, t),
                                                 p(x, y, t),
                                                 p(x+h, y, t),
                                                 p(x, y-h, t),
                                                 p(x, y+h, t), m, s, h, e),
                                                stencilA, m_sub, rec, damp, srca)

        self._gradient_stencil = GradientOperator((p(x, y, t+s),
                                                   p(x-h, y, t),
                                                   p(x, y, t),
                                                   p(x+h, y, t),
                                                   p(x, y-h, t),
                                                   p(x, y+h, t), m, s, h, e),
                                                  stencilA, u, m_sub, rec, damp)
        dm = DenseData("dm", self.model.vp.shape, self.dtype)
        dm.initializer = self.dm_initializer
        self._born_stencil = BornOperator((p(x, y, t-s),
                                           p(x-h, y, t),
                                           p(x, y, t),
                                           p(x+h, y, t),
                                           p(x, y-h, t),
                                           p(x, y+h, t), m, s, h, e),
                                          stencil, dm, m_sub, src, damp, rec)

    def Forward(self):
        self._forward_stencil.apply()
        return (self.rec.data, self.u.data)

    def Adjoint(self, rec):
        v = self._adjoint_stencil.apply()[0]
        return (self.srca.data, v)

    def Gradient(self, rec, u):
        dt = self.dt
        grad = self._gradient_stencil.apply()[0]
        return (dt**-2)*grad

    def Born(self):
        self._born_stencil.apply()
        return self.rec.data

    def run(self):
        print('Starting forward')
        rec, u = self.Forward()
        res = rec - np.transpose(self.data.traces)
        f = 0.5*np.linalg.norm(res)**2
        print('Residual is ', f, 'starting gradient')
        g = self.Gradient(res, u)
        return f, g[self.nbpml:-self.nbpml, self.nbpml:-self.nbpml]


class SourceLike(PointData):
    """Defines the behaviour of sources and receivers.
    """
    def __init__(self, name, npoint, nt, dt, h, data, dtype):
        self.orig_data = data
        self.dt = dt
        self.h = h
        super(SourceLike, self).__init__(name, npoint, nt, dtype)
        x1, z1, x2, z2 = symbols('x1, z1, x2, z2')
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
        self.rs = symbols('rx, rz')
        rx, rz = self.rs
        p = Matrix([[1],
                    [rx],
                    [rz],
                    [rx*rz]])

        self.bs = A.inv().T.dot(p)

    def point2grid(self, x, z):
        # In: s - Magnitude of the source
        #     x, z - Position of the source
        # Returns: (i, k) - Grid coordinate at top left of grid cell.
        #          (s11, s12, s21, s22) - source values at coordinates
        #          (i, k), (i, k+1), (i+1, k), (i+1, k+1)
        rx, rz = self.rs
        b11, b12, b21, b22 = self.bs
        i = int(x/self.h)
        k = int(z/self.h)

        x = x - i*self.h
        z = z - k*self.h

        s11 = b11.subs(((rx, x), (rz, z))).evalf()
        s12 = b12.subs(((rx, x), (rz, z))).evalf()
        s21 = b21.subs(((rx, x), (rz, z))).evalf()
        s22 = b22.subs(((rx, x), (rz, z))).evalf()
        return (i, k), (s11, s12, s21, s22)

    # Interpolate onto receiver point.
    def grid2point(self, u, x, z):
        rx, rz = self.rs
        b11, b12, b21, b22 = self.bs
        i = int(x/self.h)
        j = int(z/self.h)

        x = x - i*self.h
        z = z - j*self.h

        return (b11.subs(((rx, x), (rz, z))) * u[t, i, j] +
                b12.subs(((rx, x), (rz, z))) * u[t, i, j+1] +
                b21.subs(((rx, x), (rz, z))) * u[t, i+1, j] +
                b22.subs(((rx, x), (rz, z))) * u[t, i+1, j+1])

    def read(self, u):
        eqs = []
        for i in range(self.npoints):
            eqs.append(Eq(self[t, i], self.grid2point(u, self.orig_data[i, 0],
                                                      self.orig_data[i, 2])))
        return eqs

    def add(self, m, u):
        assignments = []
        dt = self.dt
        for j in range(self.npoints):
            add = self.point2grid(self.orig_data[j, 0],
                                  self.orig_data[j, 2])
            (i, k) = add[0]
            assignments.append(Eq(u[t, i, k], u[t, i, k]+self[t, j]*dt*dt/m[i, k]*add[1][0]))
            assignments.append(Eq(u[t, i, k+1], u[t, i, k+1]+self[t, j]*dt*dt/m[i, k]*add[1][1]))
            assignments.append(Eq(u[t, i+1, k], u[t, i+1, k]+self[t, j]*dt*dt/m[i, k]*add[1][2]))
            assignments.append(Eq(u[t, i+1, k+1], u[t, i+1, k+1]+self[t, j]*dt*dt/m[i, k]*add[1][3]))
        filtered = [x for x in assignments if isinstance(x, Eq)]
        return filtered

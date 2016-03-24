# coding: utf-8
from __future__ import print_function

from sympy import Function, symbols, init_printing, as_finite_diff, solve, \
                  lambdify
from sympy.abc import x, y, t, M, Q, D, E

import numpy as np
from numpy import linalg
from math import floor

from SubsurfaceModel2D import SubsurfaceModel2D
from SeismicDataReader import SeismicDataReader

init_printing()


class AcousticWave2D:
    def __init__(self, model, data, nbpml=10):
        self.model = model
        self.data = data

        self.dt = model.get_critical_dt()
        self.h = model.get_spacing()[0]
        self.nbpml = nbpml

        self._init_taylor()

    def _init_taylor(self):
        # The acoustic wave equation for the square slowness m and a source q
        # is given in 3D by :
        #
        # \begin{cases} &m \frac{d^2 u(x,t)}{dt^2} - \nabla^2 u(x,t) =q  \\
        # &u(.,0) = 0 \\ &\frac{d u(x,t)}{dt}|_{t=0} = 0 \end{cases}
        #
        # with the zero initial conditons to guarantee unicity of the solution

        p = Function('p')
        m, s, h = symbols('m s h')
        m = M(x, y)
        q = Q(x, y, t)
        d = D(x, y, t)
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
        wave_equation = m * dtt - (dxx + dyy) - q + e * dt
        stencil = solve(wave_equation, p(x, y, t+s))[0]
        self.ts = lambdify((p(x, y, t-s),
                           p(x-h, y, t),
                           p(x, y, t),
                           p(x+h, y, t),
                           p(x, y-h, t),
                           p(x, y+h, t), q, m, s, h, e),
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
        wave_equationA = m * dtt - (dxx + dyy) - D(x, y, t) - e * dt
        stencilA = solve(wave_equationA, p(x, y, t-s))[0]
        self.tsA = lambdify((p(x, y, t+s),
                            p(x-h, y, t),
                            p(x, y, t),
                            p(x+h, y, t),
                            p(x, y-h, t),
                            p(x, y+h, t), d, m, s, h, e),
                            stencilA, "numpy")

    def dampx(self, x):
        nbpml = self.nbpml
        h = self.h
        nx, ny = self.model.get_dimensions()

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
        nx, ny = self.model.get_dimensions()

        dampcoeff = 1.5 * np.log(1.0 / 0.001) / (5.0 * h)
        if y < nbpml:
            return dampcoeff*((nbpml-y)/nbpml)**2
        elif y > ny - nbpml - 1:
            return dampcoeff * ((y - ny + nbpml) / nbpml)**2
        else:
            return 0.0

    # Interpolate source onto grid...
    def source_interpolate(self, x, y):

        if (abs(x - self.data.get_source_loc()[0]) < self.h / 2 and
                abs(y - self.data.get_source_loc()[1]) < self.h / 2):
            return 1
        else:
            return 0

    def Forward(self, nt):
        xmin, ymin = self.model.get_origin()
        nx, ny = self.model.get_dimensions()
        m = self.model.get_vp()
        dt = self.dt
        h = self.h

        u = np.zeros((nt, nx, ny))
        rec = np.zeros((nt, ny - 2))
        for ti in range(0, nt):
            for a in range(1, nx-1):
                for b in range(1, ny-1):
                    if self.source_interpolate(xmin+a*h, ymin+h*b):
                        src = self.data.get_source(ti)
                    else:
                        src = 0
                    damp = self.dampx(a) + self.dampy(b)
                    if ti == 0:
                        u[ti, a, b] = self.ts(0, 0, 0, 0, 0, 0,
                                              src, m[a, b], dt, h, damp)
                    elif ti == 1:
                        u[ti, a, b] = self.ts(0, u[ti - 1, a - 1, b],
                                              u[ti - 1, a, b],
                                              u[ti - 1, a + 1, b],
                                              u[ti - 1, a, b - 1],
                                              u[ti - 1, a, b + 1],
                                              src, m[a, b], dt, h, damp)
                    else:
                        u[ti, a, b] = self.ts(u[ti - 2, a, b],
                                              u[ti - 1, a - 1, b],
                                              u[ti - 1, a, b],
                                              u[ti - 1, a + 1, b],
                                              u[ti - 1, a, b - 1],
                                              u[ti - 1, a, b + 1],
                                              src, m[a, b], dt, h, damp)
                    if a == xrec:
                        rec[ti, b - 1] = u[ti, a, b]
        return rec, u

    def Adjoint(self, nt, rec):
        xmin, ymin = self.model.get_origin()
        nx, ny = self.model.get_dimensions()
        dt = self.dt
        h = self.h
        m = self.model.get_vp()
        v = np.zeros((nt, nx, ny))
        srca = np.zeros((nt))
        for ti in range(nt - 1, -1, -1):
            for a in range(1, nx-1):
                for b in range(1, ny-1):
                    if a == xrec:
                        resid = rec[ti, b-1]
                    else:
                        resid = 0
                        damp = self.dampx(a) + self.dampy(b)
                    if ti == nt-1:
                        v[ti, a, b] = self.tsA(0, 0, 0, 0, 0, 0,
                                               resid, m[a, b], dt, h, damp)
                    elif ti == nt-2:
                        v[ti, a, b] = self.tsA(0,
                                               v[ti+1, a-1, b],
                                               v[ti+1, a, b],
                                               v[ti+1, a+1, b],
                                               v[ti+1, a, b-1],
                                               v[ti+1, a, b+1],
                                               resid, m[a, b],
                                               dt, h, damp)
                    else:
                        v[ti, a, b] = self.tsA(v[ti + 2, a, b],
                                               v[ti + 1, a - 1, b],
                                               v[ti + 1, a, b],
                                               v[ti + 1, a + 1, b],
                                               v[ti + 1, a, b - 1],
                                               v[ti + 1, a, b + 1],
                                               resid, m[a, b], dt, h, damp)
                    if self.source_interpolate(xmin+a*h, ymin+h*b):
                        srca[ti] = v[ti, a, b]
        return srca, v

    def Gradient(self, nt, rec, u):
        xmin, ymin = self.model.get_origin()
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
                    if a == xrec:
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
                    if a == xrec:
                        rec[ti, b-1] = U3[a, b]
            u1, u2, u3 = u2, u3, u1
            U1, U2, U3 = U2, U3, U1
        return rec

    def compute_gradient(self, model, shot_id):
        raise NotImplementedError("compute_gradient")

# Set up test case
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import animation

    # Define the discrete model
    model = SubsurfaceModel2D()
    model0 = SubsurfaceModel2D()
    dimensions = (100, 100)
    origin = (0., 0.)
    spacing = (25., 25)

    # Velocity models
    def smooth10(vel, nx, ny):
        out = np.ones((nx, ny))
        out[:, :] = vel[:, :]
        for a in range(5, nx-6):
            out[a, :] = np.sum(vel[a - 5:a + 5, :], axis=0) / 10
        return out

    # True velocity
    true_vp = np.ones(dimensions) + 2.0
    true_vp[floor(dimensions[0] / 2):dimensions[0], :] = 4.5

    # Smooth velocity
    initial_vp = smooth10(true_vp, dimensions[0], dimensions[1])
    dm = true_vp**-2 - initial_vp**-2
    dv = -true_vp + initial_vp
    model.create_model(origin, spacing, true_vp)
    model0.create_model(origin, spacing, initial_vp)
    # Define seismic data.
    data = SeismicDataReader()

    f0 = .010
    dt = model.get_critical_dt()
    t0 = 0.0
    tn = 1000.0
    nt = int(1+(tn-t0)/dt)

    # Set up the source as Ricker wavelet for f0
    def source(t, f0):
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)

    time_series = source(np.linspace(t0, tn, nt), f0)
    location = (origin[0] + dimensions[0] * spacing[0] * 0.5,
                origin[1] + dimensions[1] * spacing[1] * 0.05)
    data.set_source(time_series, dt, location)

    xrec = 10 + 4
    # A Forward propagation example
    (rect, ut) = AcousticWave2D(model, data).Forward(nt)

    # get ready to populate this list the Line artists to be plotted
    fig = plt.figure()
    plts = []
    plt.hold("off")
    for i in range(nt):
        # this is how you'd plot a single line...
        r = plt.imshow(ut[i, :, :])
        plts.append([r])
    # run the animation
    ani = animation.ArtistAnimation(fig, plts, interval=50,  repeat=False)
    # plt.show()

    fig2 = plt.figure()
    plt.hold("off")
    shotrec = plt.imshow(rect)   # this is how you'd plot a single line...
    # plt.show()

    # Adjoint test
    #
    # In ordr to guaranty we have the gradient we need to make sure that the
    # solution of the adjoint wave equation is indeed the true adjoint. Tod os
    # so one should check that
    #
    # $ <Ax,y> - <x,A^Ty> = 0$
    #
    # where $A$ is the wave_equation,  $A^T$ is wave_equationA and $x,y$ are
    # any random vectors in the range of each operator. This can however be
    # expensive as this two vector would be of size $N * n_t$. To test our
    # operator we will the relax this test by
    #
    # $ <P_r A P_s^T x,y> - <x,P_SA^TP_r^Ty> = 0$
    #
    # where $P_r , P_s^T$ are the source and recevier projection operator
    # mapping the source and receiver locations and times onto the full domain.
    # This allow to have only a random source of size $n_t$ at a random
    # postion.
    (rec0, u0) = AcousticWave2D(model0, data).Forward(nt)
    (srca, v) = AcousticWave2D(model0, data).Adjoint(nt, rec0)

    # get ready to populate this list the Line artists to be plotted
    plts = []
    plt.hold("off")
    for i in range(0, nt):
        # this is how you'd plot a single line...
        r = plt.imshow(v[i, :, :])
        plts.append([r])
    # run the animation
    ani = animation.ArtistAnimation(fig, plts, interval=50,  repeat=False)

    # this is how you'd plot a single line...
    shotrec = plt.plot(srca)
    # plt.show()

    # Actual adjoint test
    term1 = 0
    for ti in range(0, nt):
        term1 = term1 + srca[ti] * data.get_source(ti)

    term2 = linalg.norm(rec0)**2

    term1, term2, term1 - term2, term1 / term2

    # Least square objective Gradient
    #
    # We will consider here the least square objective, as this is the one in
    # need of an adjoint. The test that will follow are however necessary for
    # any objective and associated gradient in a optimization framework. The
    # objective function can be written
    #
    # $ min_m \Phi(m) := \frac{1}{2} \| P_r A^{-1}(m) q - d\|_2^2$
    #
    # And it's gradient becomes
    #
    # $ \nabla_m \Phi(m) = - (\frac{dA(m)u}{dm})^T v $
    #
    # where v is the soltuion if the adjoint wave equation. For the simple
    # acoustic case the  gradient can be rewritten as
    #
    # $ \nabla_m \Phi(m) = - \sum_{t=1}^{nt} \frac{d^2u(t)}{dt^2} v(t) $

    # Misfit
    F0 = .5 * linalg.norm(rec0 - rect)**2

    Im1 = AcousticWave2D(model0, data).Gradient(nt, rec0 - rect, u0)

    # this is how you'd plot a single line...
    shotrec = plt.imshow(rect, vmin=-1, vmax=1)

    # this is how you'd plot a single line...
    shotrec = plt.imshow(rec0, vmin=-1, vmax=1)

    # this is how you'd plot a single line...
    shotrec = plt.imshow(rec0 - rect, vmin=-.1, vmax=.1)

    # this is how you'd plot a single line...
    shotrec = plt.imshow(Im1, vmin=-1, vmax=1)

    # Adjoint test for the gradient
    #
    # The adjoint of the FWI Gradient is the Born modelling operator,
    # implementing a double propagation forward in time with a wavefield scaled
    # by the model perturbation for the second propagation
    #
    #  $ J dm = - A^{-1}(\frac{d A^{-1}q}{dt^2}) dm $

    Im2 = AcousticWave2D(model0, data).Gradient(nt, rec0, u0)

    du1 = AcousticWave2D(model0, data).Born(nt, dm)

    term11 = np.dot((rec0).reshape(-1), du1.reshape(-1))
    term21 = np.dot(Im2.reshape(-1), dm.reshape(-1))

    term11, term21, term11 - term21, term11 / term21

    # Jacobian test
    # The last part is to check that the operators are consistent with
    # the problem. There is then two properties to be satisfied
    #
    # $ U(m + hdm) = U(m) +  \mathcal{O} (h) \\
    #    U(m + h dm) = U(m) + h J[m]dm + \mathcal{O} (h^2) $
    #
    # which are the linearization conditions for the objective. This is
    # a bit slow to run here but here is the way to test it.
    #
    # 1 - Genrate data for the true model m
    # 2 - Define a smooth initial model $m_0$ and comput the data $d_0$
    # for this model
    # 3 - You now have $U(m_0)$
    # 4 - Define $ dm = m-m_0$ and $ h = {1,.1,.01,.001,...}$
    # 5 - For each $h$ compute $U(m_0 + h dm)$ by generating data for
    # $m_0 + h dm$ and compute $(J[m_0 + h dm]^T\delta |d) $
    # 6 - Plot in Loglog the two lines of equation above

    H = [1, 0.1, 0.01, .001, 0.0001, 0.00001, 0.000001]
    (D1, u0) = AcousticWave2D(model0, data).Forward(nt)
    dub = AcousticWave2D(model0, data).Born(nt, dm)
    error1 = np.zeros((7))
    error2 = np.zeros((7))
    for i in range(0, 7):
        mloc = initial_vp + H[i] * dv
        model0.set_vp(mloc)
        (d, u) = AcousticWave2D(model0, data).Forward(nt)
        error1[i] = linalg.norm(d - D1, ord=1)
        error2[i] = linalg.norm(d - D1 - H[i] * dub, ord=1)

    hh = np.zeros((7))
    for i in range(0, 7):
        hh[i] = H[i] * H[i]
    # this is how you'd plot a single line...
    shotrec = plt.loglog(H, error1, H, H)
    plt.show()
    # this is howyou'd plot a single line...
    shotrec = plt.loglog(H, error2, H, hh)
    plt.show()

    # Gradient test
    # The last part is to check that the operators are consistent with
    # the problem. There is then two properties to be satisfied
    #
    # $ \Phi(m + hdm) = \Phi(m) +  \mathcal{O} (h) \\
    #    \Phi(m + h dm) = \Phi(m) + h (J[m]^T\delta |d)dm + \mathcal{O} (h^2) $
    #
    # which are the linearization conditions for the objective. This is a bit
    # slow to run here but here is the way to test it.
    #
    # 1 - Genrate data for the true model m
    # 2 - Define a smooth initial model $m_0$ and comput the data $d_0$ for
    # this model
    # 3 - You now have $\Phi(m_0)$
    # 4 - Define $ dm = m-m_0$ and $ h = {1,.1,.01,.001,...}$
    # 5 - For each $h$ compute $\Phi(m_0 + h dm)$ by generating data for
    # $m_0 + h dm$ and compute $(J[m_0 + h dm]^T\delta |d) $
    # 6 - Plot in Loglog the two lines of equation above
    model0.set_vp(initial_vp)
    (DT, uT) = AcousticWave2D(model, data).Forward(nt)
    (D1, u0) = AcousticWave2D(model0, data).Forward(nt)
    F0 = .5*linalg.norm(D1 - DT)**2

    g = AcousticWave2D(model0, data).Gradient(nt, D1 - DT, u0)
    G = np.dot(g.reshape(-1), dm.reshape(-1))
    error21 = np.zeros((7))
    error22 = np.zeros((7))
    for i in range(0, 7):
        mloc = initial_vp + H[i] * dv
        model0.set_vp(mloc)
        (Dloc, u) = AcousticWave2D(model0, data).Forward(nt)
        error21[i] = .5*linalg.norm(Dloc - DT)**2 - F0
        error22[i] = .5*linalg.norm(Dloc - DT)**2 - F0 - H[i] * G

        # this is how you'd plot a single line...
    shotrec = plt.loglog(H, error21, H, H)
    plt.show()
    # this is how you'd plot a single line...
    shotrec = plt.loglog(H, error22, H, hh)
    plt.show()

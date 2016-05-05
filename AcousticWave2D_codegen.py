# coding: utf-8
from __future__ import print_function
from sympy import Function, symbols, init_printing, as_finite_diff
from sympy import solve, Eq, Matrix
from sympy.abc import x, y, t, M, Q, D, E
import numpy as np
from jit_manager import JitManager
from operators import *

init_printing()


class AcousticWave2D_cg:

    def __init__(self, model, data, nbpml=10):
        self.model = model
        self.data = data
        self.dtype = np.float64
        self.dt = model.get_critical_dt()
        self.h = model.get_spacing()
        self.nbpml = nbpml
        self.src_grid = None
        self.nt = self.data.nt
        
         # Set up interpolation from grid to receiver position.
        
    def prepare(self):
        self._init_taylor(self.nt)
        self._forward_stencil, self._adjoint_stencil, self._gradient_stencil, self._born_stencil = self.jit_manager.get_wrapped_functions()

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
        wave_equation = m * dtt - (dxx + dyy) + e * dt
        stencil = solve(wave_equation, p(x, y, t+s))[0]
        self.nt = nt

        prop_fw = ForwardOperator((p(x, y, t-s),
                           p(x-h, y, t),
                           p(x, y, t),
                           p(x+h, y, t),
                           p(x, y-h, t),
                           p(x, y+h, t), m, s, h, e),
                           stencil, self).get_propagator()
        fds = [prop_fw]

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
        wave_equationA = m * dtt - (dxx + dyy) - e * dt
        stencilA = solve(wave_equationA, p(x, y, t-s))[0]
        prop_adj = AdjointOperator((p(x, y, t+s),
                            p(x-h, y, t),
                            p(x, y, t),
                            p(x+h, y, t),
                            p(x, y-h, t),
                            p(x, y+h, t), m, s, h, e),
                            stencilA, self).get_propagator()

        fds.append(prop_adj)

        prop_grad = GradientOperator((p(x, y, t+s),
                            p(x-h, y, t),
                            p(x, y, t),
                            p(x+h, y, t),
                            p(x, y-h, t),
                            p(x, y+h, t), m, s, h, e),
                            stencilA, self).get_propagator()
        fds.append(prop_grad)

        prop_born = BornOperator((p(x, y, t-s),
                                             p(x-h, y, t),
                                             p(x, y, t),
                                             p(x+h, y, t),
                                             p(x, y-h, t),
                                             p(x, y+h, t), m, s, h, e),
                                            stencil, self).get_propagator()
        fds.append(prop_born)
        self.jit_manager = JitManager(fds, dtype=self.dtype)

    def Forward(self):
        nx, ny = self.model.get_shape()
        m = self.model.vp**(-2)
        nt = self.nt
        u = np.zeros((nt+2, nx, ny), dtype=self.dtype)
        rec = np.zeros((nt, ny - 2), dtype=self.dtype)
        self._forward_stencil(u, rec, m, self.data.get_source(),
                              self.dampx, self.dampy)
        return rec, u

    def Adjoint(self, rec):
        nt = self.nt
        nx, ny = self.model.get_shape()
        m = self.model.vp**(-2)
        v = np.zeros((nt+2, nx, ny))
        srca = np.zeros((nt))
        self._adjoint_stencil(v, rec, m, srca,
                              self.dampx, self.dampy)
        return srca, v

    def Gradient(self, rec, u):
        nx, ny = self.model.get_shape()
        dt = self.dt
        m = self.model.vp**(-2)
        v = np.zeros((3, nx, ny))
        grad = np.zeros((nx, ny))
        self._gradient_stencil(u, rec, v, grad, m, self.dampx, self.dampy)
        return (dt**-2)*grad

    def Born(self, dm):
        nt = self.nt
        nx, ny = self.model.get_shape()
        m = self.model.vp**(-2)
        u = np.zeros((3, nx, ny))
        U = np.zeros((3, nx, ny))
        rec = np.zeros((nt, ny-2))
        self._born_stencil(u, U, rec, dm, m, self.dampx, self.dampy, self.data.get_source())
        return rec

    def source_interpolate(self):
        if self.src_grid is not None:
            return self.src_grid
        xmin, ymin = self.model.get_origin()
        nx, ny = self.model.get_shape()
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

    # Interpolate source onto grid.
    

    def compute_gradient(self, model, shot_id):
        raise NotImplementedError("compute_gradient")

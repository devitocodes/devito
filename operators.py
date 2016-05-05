from propagator import Propagator
from sympy.abc import x, y ,t
from sympy import Eq, symbols, Matrix


class Operator(object):
    def __init__(self, subs, stencil, problem):
        self.subs = subs
        self.stencil = stencil
        self.problem = problem
        self.nt = problem.nt
        self.shape = problem.model.get_shape()
        self.dtype = problem.dtype
        self.dt = problem.dt
        self.h = problem.h
        self.data = problem.data
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
        self.rs = symbols('rx, rz')
        rx, rz = self.rs
        p = Matrix([[1],
                          [rx],
                          [rz],
                          [rx*rz]])

        self.bs = A.inv().T.dot(p)

    def get_propagator(self):
        return self._prepare(self.subs, self.stencil, self.nt, self.shape)
    
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

class ForwardOperator(Operator):
    def _prepare(self, subs, stencil, nt, shape):
        nx, ny = shape
        propagator = Propagator("Forward", nt, (nx, ny), spc_border=1, time_order=2)
        u = propagator.add_param("u", (nt, nx, ny), self.dtype)
        rec = propagator.add_param("rec", (nt, ny - 2), self.dtype)
        M = propagator.add_param("m", (nx, ny), self.dtype)
        src_time = propagator.add_param("src_time", (nt,), self.dtype)
        dampx = propagator.add_param("dampx", (nx,), self.dtype)
        dampy = propagator.add_param("dampy", (nx,), self.dtype)

        stencil_args = [u[t - 2, x, y],
                          u[t - 1, x - 1, y],
                          u[t - 1, x, y],
                          u[t - 1, x + 1, y],
                          u[t - 1, x, y - 1],
                          u[t - 1, x, y + 1],
                          M[x, y], self.dt, self.h, dampx[x]+dampy[y]]
        lhs = u[t, x, y]
        main_stencil = Eq(lhs, stencil)
        propagator.set_jit_params(subs, [main_stencil], [stencil_args])
        src = self.add_source(src_time, M, self.dt, u)
        for sten in src: propagator.add_time_loop_stencil(sten)
        rec = self.read_rec(rec, u)
        for sten in rec: propagator.add_time_loop_stencil(sten)
        return propagator
    
    def add_source(self, src, m, dt, u):
        src_add = self.point2grid(self.data.source_coords[0],
                                  self.data.source_coords[2])
        (i, k) = src_add[0]
        weights = src_add[1]
        assignments = []
        assignments.append(Eq(u[t, i, k], u[t, i, k]+src[t-2]*dt*dt/m[i, k]*weights[0]))
        assignments.append(Eq(u[t, i, k+1], u[t, i, k+1]+src[t-2]*dt*dt/m[i, k]*weights[1]))
        assignments.append(Eq(u[t, i+1, k], u[t, i+1, k]+src[t-2]*dt*dt/m[i, k]*weights[2]))
        assignments.append(Eq(u[t, i+1, k+1], u[t, i+1, k+1]+src[t-2]*dt*dt/m[i, k]*weights[3]))
        filtered = [x for x in assignments if isinstance(x, Eq)]
        return filtered
    
    def read_rec(self, rec, u):
        ntraces, nsamples = self.data.traces.shape
        eqs = []
        for i in range(ntraces):
            eqs.append(Eq(rec[t-2, i], self.grid2point(u, self.data.receiver_coords[i, 0],
                                     self.data.receiver_coords[i, 2])))
        return eqs
    

class AdjointOperator(Operator):
    def _prepare(self, subs, stencil, nt, shape):
        nx, ny = shape
        propagator = Propagator("Adjoint", nt, (nx, ny), spc_border=1, forward=False, time_order=2)
        v = propagator.add_param("v", (nt, nx, ny), self.dtype)
        rec = propagator.add_param("rec", (nt, ny - 2), self.dtype)
        M = propagator.add_param("m", (nx, ny), self.dtype)
        srca = propagator.add_param("srca", (nt,), self.dtype)
        dampx = propagator.add_param("dampx", (nx,), self.dtype)
        dampy = propagator.add_param("dampy", (nx,), self.dtype)
        dt = self.dt
        h = self.h
        lhs = v[t, x, y]
        main_stencil = Eq(lhs, stencil)
        stencil_args = [v[t + 2, x, y],
                       v[t + 1, x - 1, y],
                       v[t + 1, x, y],
                       v[t + 1, x + 1, y],
                       v[t + 1, x, y - 1],
                       v[t + 1, x, y + 1],
                       M[x, y], dt, h, dampx[x]+dampy[y]]
        propagator.set_jit_params(subs, [main_stencil], [stencil_args])
        rec = self.add_rec(rec, M, self.dt, v)
        for sten in rec: propagator.add_time_loop_stencil(sten)
        src = self.read_source(srca, v)
        for sten in src: propagator.add_time_loop_stencil(sten)
        return propagator
    
    def add_rec(self, rec, m, dt, u):
        ntraces, nsamples = self.data.traces.shape
        assignments = []
        for j in range(ntraces):
            rec_add = self.point2grid(self.data.receiver_coords[j, 0],
                                      self.data.receiver_coords[j, 2])
            (i, k) = rec_add[0]
            
            assignments.append(Eq(u[t, i, k], u[t, i, k]+rec[t, j]*dt*dt/m[i, k]*rec_add[1][0]))
            assignments.append(Eq(u[t, i, k+1], u[t, i, k+1]+rec[t, j]*dt*dt/m[i, k]*rec_add[1][1]))
            assignments.append(Eq(u[t, i+1, k], u[t, i+1, k]+rec[t, j]*dt*dt/m[i, k]*rec_add[1][2]))
            assignments.append(Eq(u[t, i+1, k+1], u[t, i+1, k+1]+rec[t, j]*dt*dt/m[i, k]*rec_add[1][3]))
        filtered = [x for x in assignments if isinstance(x, Eq)]
        return filtered
    
    def read_source(self, src, u):
        return [Eq(src[t], self.grid2point(u, self.data.source_coords[0],
                               self.data.source_coords[2]))]

class GradientOperator(AdjointOperator):
    def _prepare(self, subs, stencil, nt, shape):
        nx, ny = shape
        propagator = Propagator("Gradient", nt, (nx, ny), spc_border=1, forward=False, time_order=2)
        u = propagator.add_param("u", (nt, nx, ny), self.dtype)
        rec = propagator.add_param("rec", (nt, ny - 2), self.dtype)
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
                             grad[x, y] - (v[t, x, y] - 2 * v[t + 1, x, y] + v[t + 2, x, y]) * (u[t, x, y]))
        propagator.set_jit_params(subs, [main_stencil, gradient_update], [stencil_args, []])
        propagator.add_loop_step(Eq(v[t+2, x, y], 0), False)
        rec = self.add_rec(rec, M, self.dt, v)
        for sten in rec: propagator.add_time_loop_stencil(sten, before=True)
        return propagator


class BornOperator(ForwardOperator):
    def _prepare(self, subs, stencil, nt, shape):
        nx, ny = shape
        propagator = Propagator("Born", nt, (nx, ny), spc_border=1, time_order=2)
        u = propagator.add_param("u", (nt, nx, ny), self.dtype, save=False)
        U = propagator.add_param("U", (nt, nx, ny), self.dtype, save=False)
        rec = propagator.add_param("rec", (nt, ny - 2), self.dtype)
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
        for sten in src: propagator.add_time_loop_stencil(sten, before=True)
        rec = self.read_rec(rec, U)
        for sten in rec: propagator.add_time_loop_stencil(sten, before=False)
        first_stencil_args = [u[t-2, x, y],
                              u[t-1, x - 1, y],
                              u[t-1, x, y],
                              u[t-1, x + 1, y],
                              u[t-1, x, y - 1],
                              u[t-1, x, y + 1],
                              M[x, y], dt, h, dampx[x] + dampy[y]]
        first_update = Eq(u[t, x, y], u[t, x, y]+stencil)
        calc_src = Eq(src2, -(dt**-2)*(u[t, x, y]-2*u[t-1, x, y]+u[t-2, x, y])*dm[x, y])
        second_stencil_args = [U[t-2, x, y],
                               U[t-1, x - 1, y],
                               U[t-1, x, y],
                               U[t-1, x + 1, y],
                               U[t-1, x, y - 1],
                               U[t-1, x, y + 1],
                               M[x, y], dt, h, dampx[x] + dampy[y]]
        second_update = Eq(U[t, x, y], stencil)
        insert_second_source = Eq(U[t, x, y], U[t, x, y]+(dt*dt)/M[x, y]*src2)
        propagator.set_jit_params(subs, [first_update, calc_src, second_update, insert_second_source], [first_stencil_args, [], second_stencil_args, []])
        return propagator
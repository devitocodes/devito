from sympy import Eq, Matrix, solve, symbols
from sympy.abc import t

from devito.interfaces import DenseData, PointData, TimeData
from devito.operator import *


class SourceLike(PointData):
    """Defines the behaviour of sources and receivers.
    """
    def __init__(self, *args, **kwargs):
        self.dt = kwargs.get('dt')
        self.h = kwargs.get('h')
        self.ndim = kwargs.get('ndim')
        self.nbpml = kwargs.get('nbpml')
        PointData.__init__(self, *args, **kwargs)
        x1, y1, z1, x2, y2, z2 = symbols('x1, y1, z1, x2, y2, z2')

        if self.ndim == 2:
            A = Matrix([[1, x1, z1, x1*z1],
                        [1, x1, z2, x1*z2],
                        [1, x2, z1, x2*z1],
                        [1, x2, z2, x2*z2]])
            self.increments = (0, 0), (0, 1), (1, 0), (1, 1)
            self.rs = symbols('rx, rz')
            rx, rz = self.rs
            p = Matrix([[1],
                        [rx],
                        [rz],
                        [rx*rz]])
        else:
            A = Matrix([[1, x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1],
                        [1, x1, y2, z1, x1*y2, x1*z1, y2*z1, x1*y2*z1],
                        [1, x2, y1, z1, x2*y1, x2*z1, y2*z1, x2*y1*z1],
                        [1, x1, y1, z2, x1*y1, x1*z2, y1*z2, x1*y1*z2],
                        [1, x2, y2, z1, x2*y2, x2*z1, y2*z1, x2*y2*z1],
                        [1, x1, y2, z2, x1*y2, x1*z2, y2*z2, x1*y2*z2],
                        [1, x2, y1, z2, x2*y1, x2*z2, y1*z2, x2*y1*z2],
                        [1, x2, y2, z2, x2*y2, x2*z2, y2*z2, x2*y2*z2]])
            self.increments = (0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)
            self.rs = symbols('rx, ry, rz')
            rx, ry, rz = self.rs
            p = Matrix([[1],
                        [rx],
                        [ry],
                        [rz],
                        [rx*ry],
                        [rx*rz],
                        [ry*rz],
                        [rx*ry*rz]])

        # Map to reference cell
        reference_cell = [(x1, 0),
                          (y1, 0),
                          (z1, 0),
                          (x2, self.h),
                          (y2, self.h),
                          (z2, self.h)]
        A = A.subs(reference_cell)
        self.bs = A.inv().T.dot(p)

    def point2grid(self, pt_coords):
        # In: s - Magnitude of the source
        #     x, z - Position of the source
        # Returns: (i, k) - Grid coordinate at top left of grid cell.
        #          (s11, s12, s21, s22) - source values at coordinates
        #          (i, k), (i, k+1), (i+1, k), (i+1, k+1)
        if self.ndim == 2:
            rx, rz = self.rs
        else:
            rx, ry, rz = self.rs

        x, y, z = pt_coords
        i = int(x/self.h)
        k = int(z/self.h)
        coords = (i + self.nbpml, k + self.nbpml)
        subs = []
        x = x - i*self.h
        subs.append((rx, x))

        if self.ndim == 3:
            j = int(y/self.h)
            y = y - j*self.h
            subs.append((ry, y))
            coords = (i + self.nbpml, j + self.nbpml, k + self.nbpml)

        z = z - k*self.h
        subs.append((rz, z))
        s = [b.subs(subs).evalf() for b in self.bs]

        return coords, tuple(s)

    # Interpolate onto receiver point.
    def grid2point(self, u, pt_coords):
        if self.ndim == 2:
            rx, rz = self.rs
        else:
            rx, ry, rz = self.rs

        x, y, z = pt_coords
        i = int(x/self.h)
        k = int(z/self.h)

        x = x - i*self.h
        z = z - k*self.h

        subs = []
        subs.append((rx, x))

        if self.ndim == 3:
            j = int(y/self.h)
            y = y - j*self.h
            subs.append((ry, y))

        subs.append((rz, z))

        if self.ndim == 2:
            return sum(
                [b.subs(subs) * u.indexed[t, i+inc[0]+self.nbpml, k+inc[1]+self.nbpml]
                    for inc, b in zip(self.increments, self.bs)])
        else:
            return sum(
                [b.subs(subs) * u.indexed[t, i+inc[0]+self.nbpml, j+inc[1]+self.nbpml, k+inc[2]+self.nbpml]
                    for inc, b in zip(self.increments, self.bs)])

    def read(self, u):
        eqs = []

        for i in range(self.npoint):
            eqs.append(Eq(self.indexed[t, i], self.grid2point(u, self.coordinates[i, :])))
        return eqs

    def add(self, m, u):
        assignments = []
        dt = self.dt

        for j in range(self.npoint):
            add = self.point2grid(self.coordinates[j, :])
            coords = add[0]
            s = add[1]
            assignments += [Eq(u.indexed[tuple([t] + [coords[i] + inc[i] for i in range(self.ndim)])],
                               u.indexed[tuple([t] + [coords[i] + inc[i] for i in range(self.ndim)])] +
                               self.indexed[t, j]*dt*dt/m.indexed[coords]*w) for w, inc in zip(s, self.increments)]

        filtered = [x for x in assignments if isinstance(x, Eq)]

        return filtered


class ForwardOperator(Operator):
    def __init__(self, m, src, damp, rec, u, time_order=2, spc_order=6, **kwargs):
        assert(m.shape == damp.shape)

        u.pad_time = False

        # Set time and space orders
        u.time_order = time_order
        u.space_order = spc_order

        # Derive stencil from symbolic equation
        eqn = m * u.dt2 - u.laplace + damp * u.dt
        stencil = solve(eqn, u.forward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: src.dt, h: src.h}

        super(ForwardOperator, self).__init__(src.nt, m.shape, stencils=Eq(u.forward, stencil),
                                              substitutions=subs, spc_border=spc_order/2,
                                              time_order=time_order, forward=True, dtype=m.dtype,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [src, rec]
        self.propagator.time_loop_stencils_a = src.add(m, u) + rec.read(u)
        self.propagator.add_devito_param(src)
        self.propagator.add_devito_param(rec)


class AdjointOperator(Operator):
    def __init__(self, m, rec, damp, srca, time_order=2, spc_order=6, **kwargs):
        assert(m.shape == damp.shape)

        # Create v with given time and space orders
        v = TimeData(name="v", shape=m.shape, dtype=m.dtype, time_dim=rec.nt,
                     time_order=time_order, space_order=spc_order, save=True)

        # Derive stencil from symbolic equation
        eqn = m * v.dt2 - v.laplace - damp * v.dt
        stencil = solve(eqn, v.backward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: rec.dt, h: rec.h}

        # Input/output signature detection is still dubious,
        # so we need to keep this hard-coded for now
        input_params = [m, rec, damp, srca]
        output_params = [v]

        super(AdjointOperator, self).__init__(rec.nt, m.shape, stencils=Eq(v.backward, stencil),
                                              substitutions=subs, spc_border=spc_order/2,
                                              time_order=time_order, forward=False, dtype=m.dtype,
                                              input_params=input_params, output_params=output_params,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_a = rec.add(m, v) + srca.read(v)


class GradientOperator(Operator):
    def __init__(self, u, m, rec, damp, time_order=2, spc_order=6, **kwargs):
        assert(m.shape == damp.shape)

        v = TimeData(name="v", shape=m.shape, dtype=m.dtype, time_dim=rec.nt,
                     time_order=time_order, space_order=spc_order, save=False, )
        grad = DenseData(name="grad", shape=m.shape, dtype=m.dtype)

        # Derive stencil from symbolic equation
        eqn = m * v.dt2 - v.laplace - damp * v.dt
        stencil = solve(eqn, v.backward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: rec.dt, h: rec.h}

        # Add Gradient-specific updates and resets
        total_dim = tuple(v.indices(m.shape))
        space_dim = tuple(m.indices(m.shape))
        gradient_update = Eq(grad.indexed[space_dim], grad.indexed[space_dim] -
                             (v.indexed[total_dim] - 2 * v.indexed[tuple((t + 1,) + space_dim)] +
                              v.indexed[tuple((t + 2,) + space_dim)]) * u.indexed[total_dim])
        reset_v = Eq(v.indexed[tuple((t + 2,) + space_dim)], 0)
        stencils = [Eq(v.backward, stencil), gradient_update, reset_v]

        # Input/output signature detection is still dubious,
        # so we need to keep this hard-coded for now
        input_params = [u, m, rec, damp]
        output_params = [grad, v]

        super(GradientOperator, self).__init__(rec.nt, m.shape, stencils=stencils,
                                               substitutions=[subs, {}, {}], spc_border=spc_order/2,
                                               time_order=time_order, forward=False, dtype=m.dtype,
                                               input_params=input_params, output_params=output_params,
                                               **kwargs)

        # Insert receiver term post-hoc
        self.propagator.time_loop_stencils_b = rec.add(m, v)


class BornOperator(Operator):
    def __init__(self, dm, m, src, damp, rec, time_order=2, spc_order=6, **kwargs):
        assert(m.shape == damp.shape)

        u = TimeData(name="u", shape=m.shape, dtype=m.dtype, time_dim=src.nt,
                     time_order=time_order, space_order=spc_order, save=False)
        U = TimeData(name="U", shape=m.shape, dtype=m.dtype, time_dim=src.nt,
                     time_order=time_order, space_order=spc_order, save=False)

        # Derive stencils from symbolic equation
        first_eqn = m * u.dt2 - u.laplace - damp * u.dt
        first_stencil = solve(first_eqn, u.forward)[0]
        second_eqn = m * U.dt2 - U.laplace - damp * U.dt
        second_stencil = solve(second_eqn, U.forward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: src.dt, h: src.h}

        # Add Born-specific updates and resets
        total_dim = tuple(u.indices(m.shape))
        space_dim = tuple(m.indices(m.shape))
        src2 = -(src.dt**-2) * (u.indexed[total_dim]-2*u.indexed[tuple((t - 1,) + space_dim)]
                                + u.indexed[tuple((t - 2,) + space_dim)]) * dm.indexed[space_dim]
        insert_second_source = Eq(U.indexed[total_dim], U.indexed[total_dim] + (src.dt * src.dt)
                                  / m.indexed[space_dim]*src2)
        reset_u = Eq(u.indexed[tuple((t - 2,) + space_dim)], 0)
        stencils = [Eq(u.forward, first_stencil), Eq(U.forward, second_stencil),
                    insert_second_source, reset_u]

        # Input/output signature detection is still dubious,
        # so we need to keep this hard-coded for now
        input_params = [dm, m, src, damp, rec]
        output_params = [u, U]

        super(BornOperator, self).__init__(src.nt, m.shape, stencils=stencils,
                                           substitutions=[subs, subs, {}, {}], spc_border=spc_order/2,
                                           time_order=time_order, forward=True, dtype=m.dtype,
                                           input_params=input_params, output_params=output_params,
                                           **kwargs)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_b = src.add(m, u)
        self.propagator.time_loop_stencils_a = rec.read(U)

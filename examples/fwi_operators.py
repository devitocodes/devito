from sympy import Eq, Function, Matrix, solve, symbols
from sympy.abc import p, t

from devito.interfaces import DenseData, PointData, TimeData
from devito.iteration import Iteration
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

    @property
    def sym_coordinates(self):
        """Symbol representing the coordinate values in each dimension"""
        return tuple([self.coordinates.indexed[p, i]
                      for i in range(self.ndim)])

    @property
    def sym_coord_indices(self):
        """Symbol for each grid index according to the coordinates"""
        return tuple([Function('INT')(Function('floor')(x / self.h))
                      for x in self.sym_coordinates])

    @property
    def sym_coord_bases(self):
        """Symbol for the base coordinates of the reference grid point"""
        return tuple([Function('FLOAT')(x - idx * self.h)
                      for x, idx in zip(self.sym_coordinates,
                                        self.sym_coord_indices)])

    def point2grid(self, u, m):
        """Generates an expression for generic point-to-grid interpolation"""
        dt = self.dt
        subs = dict(zip(self.rs, self.sym_coord_bases))
        index_matrix = [tuple([idx + ii + self.nbpml for ii, idx
                               in zip(inc, self.sym_coord_indices)])
                        for inc in self.increments]
        eqns = [Eq(u.indexed[(t, ) + idx], u.indexed[(t, ) + idx]
                   + self.indexed[t, p] * dt * dt / m.indexed[idx] * b.subs(subs))
                for idx, b in zip(index_matrix, self.bs)]
        return eqns

    def grid2point(self, u):
        """Generates an expression for generic grid-to-point interpolation"""
        subs = dict(zip(self.rs, self.sym_coord_bases))
        index_matrix = [tuple([idx + ii + self.nbpml for ii, idx
                               in zip(inc, self.sym_coord_indices)])
                        for inc in self.increments]
        return sum([b.subs(subs) * u.indexed[(t, ) + idx]
                    for idx, b in zip(index_matrix, self.bs)])

    def read(self, u):
        """Iteration loop over points performing grid-to-point interpolation."""
        interp_expr = Eq(self.indexed[t, p], self.grid2point(u))
        return [Iteration(interp_expr, variable=p, limits=self.shape[1])]

    def add(self, m, u):
        """Iteration loop over points performing point-to-grid interpolation."""
        return [Iteration(self.point2grid(u, m), variable=p, limits=self.shape[1])]


class ForwardOperator(Operator):
    def __init__(self, m, src, damp, rec, u, time_order=2, spc_order=6, **kwargs):
        assert(m.shape == damp.shape)
        u.pad_time = True
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
        self.input_params += [src, src.coordinates, rec, rec.coordinates]
        self.propagator.time_loop_stencils_a = src.add(m, u) + rec.read(u)
        self.propagator.add_devito_param(src)
        self.propagator.add_devito_param(src.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


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
        super(AdjointOperator, self).__init__(rec.nt, m.shape, stencils=Eq(v.backward, stencil),
                                              substitutions=subs, spc_border=spc_order/2,
                                              time_order=time_order, forward=False, dtype=m.dtype,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [srca, srca.coordinates, rec, rec.coordinates]
        self.propagator.time_loop_stencils_a = rec.add(m, v) + srca.read(v)
        self.propagator.add_devito_param(srca)
        self.propagator.add_devito_param(srca.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


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
        gradient_update = Eq(grad, grad - (v - 2 * v.forward + v.forward.forward) * u)
        # reset_v = Eq(v.indexed[tuple((t + 2,) + space_dim)], 0)
        stencils = [Eq(v.backward, stencil), gradient_update]
        super(GradientOperator, self).__init__(rec.nt, m.shape, stencils=stencils,
                                               substitutions=[subs, subs, {}], spc_border=spc_order/2,
                                               time_order=time_order, forward=False, dtype=m.dtype,
                                               **kwargs)

        # Insert receiver term post-hoc
        self.input_params += [u, rec, rec.coordinates]
        self.output_params = [grad]
        self.propagator.time_loop_stencils_a = rec.add(m, v)
        self.propagator.add_devito_param(u)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
        self.propagator.add_devito_param(grad)


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
        super(BornOperator, self).__init__(src.nt, m.shape, stencils=stencils,
                                           substitutions=[subs, subs, {}, {}], spc_border=spc_order/2,
                                           time_order=time_order, forward=True, dtype=m.dtype,
                                           **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [dm, src, src.coordinates, rec, rec.coordinates]
        self.output_params += [U]
        self.propagator.time_loop_stencils_b = src.add(m, u)
        self.propagator.time_loop_stencils_a = rec.read(U)
        self.propagator.add_devito_param(dm)
        self.propagator.add_devito_param(src)
        self.propagator.add_devito_param(src.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
        self.propagator.add_devito_param(U)

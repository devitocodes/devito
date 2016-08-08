from sympy import *
from sympy.abc import *

from devito.finite_difference import first_derivative
from devito.interfaces import DenseData, PointData, TimeData
from devito.iteration import Iteration
from devito.operator import *


class SourceLikeTTI(PointData):
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

    def read(self, u, v):
        """Iteration loop over points performing grid-to-point interpolation."""
        interp_expr = Eq(self.indexed[t, p], self.grid2point(u) + self.grid2point(v))
        return [Iteration(interp_expr, variable=p, limits=self.shape[1])]

    def add(self, m, u):
        """Iteration loop over points performing point-to-grid interpolation."""
        return [Iteration(self.point2grid(u, m), variable=p, limits=self.shape[1])]


class ForwardOperator(Operator):
    def __init__(self, m, src, damp, rec, u, v, A, B, th, ph, time_order=2, spc_order=4, **kwargs):
        def Bhaskarasin(angle):
            if angle == 0:
                return 0
            else:
                return 16.0 * angle * (3.1416 - abs(angle)) / (49.3483 - 4.0 * abs(angle) * (3.1416 - abs(angle)))

        def Bhaskaracos(angle):
            if angle == 0:
                return 1.0
            else:
                return Bhaskarasin(angle + 1.5708)

        Hp, Hzr = symbols('Hp Hzr')
        if len(m.shape) == 3:
            ang0 = Function('ang0')(x, y, z)
            ang1 = Function('ang1')(x, y, z)
            ang2 = Function('ang2')(x, y, z)
            ang3 = Function('ang3')(x, y, z)
        else:
            ang0 = Function('ang0')(x, y)
            ang1 = Function('ang1')(x, y)
        assert(m.shape == damp.shape)
        u.pad_time = False
        v.pad_time = False

        # Set time and space orders
        u.time_order = time_order
        u.space_order = spc_order
        v.time_order = time_order
        v.space_order = spc_order
        s, h = symbols('s h')

        ang0 = Bhaskaracos(th)
        ang1 = Bhaskarasin(th)
        # Derive stencil from symbolic equation
        if len(m.shape) == 3:
            ang2 = Bhaskaracos(ph)
            ang3 = Bhaskarasin(ph)

            Gy1p = (ang3 * u.dxl - ang2 * u.dyl)
            Gyy1 = (first_derivative(Gy1p, ang3, dim=x, side=1, order=spc_order/2) -
                    first_derivative(Gy1p, ang2, dim=y, side=1, order=spc_order/2))

            Gy2p = (ang3 * u.dxr - ang2 * u.dyr)
            Gyy2 = (first_derivative(Gy2p, ang3, dim=x, side=-1, order=spc_order/2) -
                    first_derivative(Gy2p, ang2, dim=y, side=-1, order=spc_order/2))

            Gx1p = (ang0 * ang2 * u.dxl + ang0 * ang3 * u.dyl - ang1 * u.dzl)
            Gz1r = (ang1 * ang2 * v.dxl + ang1 * ang3 * v.dyl + ang0 * v.dzl)
            Gxx1 = (first_derivative(Gx1p, ang0, ang2, dim=x, side=1, order=spc_order/2) +
                    first_derivative(Gx1p, ang0, ang3, dim=y, side=1, order=spc_order/2) -
                    first_derivative(Gx1p, ang1, dim=z, side=1, order=spc_order/2))
            Gzz1 = (first_derivative(Gz1r, ang1, ang2, dim=x, side=1, order=spc_order/2) +
                    first_derivative(Gz1r, ang1, ang3, dim=y, side=1, order=spc_order/2) +
                    first_derivative(Gz1r, ang0, dim=z, side=1, order=spc_order/2))

            Gx2p = (ang0 * ang2 * u.dxr + ang0 * ang3 * u.dyr - ang1 * u.dzr)
            Gz2r = (ang1 * ang2 * v.dxr + ang1 * ang3 * v.dyr + ang0 * v.dzr)

            Gxx2 = (first_derivative(Gx2p, ang0, ang2, dim=x, side=-1, order=spc_order/2) +
                    first_derivative(Gx2p, ang0, ang3, dim=y, side=-1, order=spc_order/2) -
                    first_derivative(Gx2p, ang1, dim=z, side=-1, order=spc_order/2))
            Gzz2 = (first_derivative(Gz2r, ang1, ang2, dim=x, side=-1, order=spc_order/2) +
                    first_derivative(Gz2r, ang1, ang3, dim=y, side=-1, order=spc_order/2) +
                    first_derivative(Gz2r, ang0, dim=z, side=-1, order=spc_order/2))
            parm = [m, damp, A, B, th, ph, u, v]
        else:
            Gyy2 = 0
            Gyy1 = 0
            parm = [m, damp, A, B, th, u, v]
            Gx1p = (ang0 * u.dxl - ang1 * u.dyl)
            Gz1r = (ang1 * v.dxl + ang0 * v.dyl)
            Gxx1 = (first_derivative(Gx1p * ang0, dim=x, side=1, order=spc_order/2) -
                    first_derivative(Gx1p * ang1, dim=y, side=1, order=spc_order/2))
            Gzz1 = (first_derivative(Gz1r * ang1, dim=x, side=1, order=spc_order/2) +
                    first_derivative(Gz1r * ang0, dim=y, side=1, order=spc_order/2))
            Gx2p = (ang0 * u.dxr - ang1 * u.dyr)
            Gz2r = (ang1 * v.dxr + ang0 * v.dyr)
            Gxx2 = (first_derivative(Gx2p * ang0, dim=x, side=-1, order=spc_order/2) -
                    first_derivative(Gx2p * ang1, dim=y, side=-1, order=spc_order/2))
            Gzz2 = (first_derivative(Gz2r * ang1, dim=x, side=-1, order=spc_order/2) +
                    first_derivative(Gz2r * ang0, dim=y, side=-1, order=spc_order/2))

        stencilp = 1.0 / (2.0 * m + s * damp) * (4.0 * m * u + (s * damp - 2.0 * m) * u.backward +
                                                 2.0 * s**2 * (A * Hp + B * Hzr))
        stencilr = 1.0 / (2.0 * m + s * damp) * (4.0 * m * v + (s * damp - 2.0 * m) * v.backward +
                                                 2.0 * s**2 * (B * Hp + Hzr))
        Hp = -(.5 * Gxx1 + .5 * Gxx2 + .5 * Gyy1 + .5 * Gyy2)
        Hzr = -(.5 * Gzz1 + .5 * Gzz2)
        factorized = {"Hp": Hp, "Hzr": Hzr}
        # Add substitutions for spacing (temporal and spatial)
        subs = [{s: src.dt, h: src.h}, {s: src.dt, h: src.h}]
        first_stencil = Eq(u.forward, stencilp)
        second_stencil = Eq(v.forward, stencilr)
        stencils = [first_stencil, second_stencil]
        super(ForwardOperator, self).__init__(src.nt, m.shape, stencils=stencils, substitutions=subs,
                                              spc_border=spc_order/2, time_order=time_order, forward=True,
                                              dtype=m.dtype,
                                              input_params=parm, factorized=factorized, **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [src, src.coordinates, rec, rec.coordinates]
        self.propagator.time_loop_stencils_a = src.add(m, u) + src.add(m, v) + rec.read(u, v)
        self.propagator.add_devito_param(src)
        self.propagator.add_devito_param(src.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)        


class AdjointOperator(Operator):
    def __init__(self, m, rec, damp, srca, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)

        input_params = [m, rec, damp, srca]
        v = TimeData("v", m.shape, rec.nt, time_order=time_order, save=True, dtype=m.dtype)
        output_params = [v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v.indexed[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'), v, fw=False)
        main_stencil = Eq(lhs, stencil)
        stencil_args = [m.indexed[space_dim], rec.dt, rec.h, damp.indexed[space_dim]]
        stencils = [main_stencil]
        substitutions = [dict(zip(subs, stencil_args))]

        super(AdjointOperator, self).__init__(rec.nt, m.shape, stencils=stencils,
                                              substitutions=substitutions, spc_border=spc_order/2,
                                              time_order=time_order, forward=False, dtype=m.dtype,
                                              input_params=input_params, output_params=output_params)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_a = rec.add(m, v) + srca.read(v)


class GradientOperator(Operator):
    def __init__(self, u, m, rec, damp, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)

        input_params = [u, m, rec, damp]
        v = TimeData("v", m.shape, rec.nt, time_order=time_order, save=False, dtype=m.dtype)
        grad = DenseData("grad", m.shape, dtype=m.dtype)
        output_params = [grad, v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v.indexed[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'), v, fw=False)
        stencil_args = [m.indexed[space_dim], rec.dt, rec.h, damp.indexed[space_dim]]
        main_stencil = Eq(lhs, lhs + stencil)
        gradient_update = Eq(grad.indexed[space_dim],
                             grad.indexed[space_dim] -
                             (v.indexed[total_dim] - 2 * v.indexed[tuple((t + 1,) + space_dim)] +
                                 v.indexed[tuple((t + 2,) + space_dim)]) * u.indexed[total_dim])
        reset_v = Eq(v.indexed[tuple((t + 2,) + space_dim)], 0)
        stencils = [main_stencil, gradient_update, reset_v]
        substitutions = [dict(zip(subs, stencil_args)), {}, {}]

        super(GradientOperator, self).__init__(rec.nt, m.shape, stencils=stencils,
                                               substitutions=substitutions, spc_border=spc_order/2,
                                               time_order=time_order, forward=False, dtype=m.dtype,
                                               input_params=input_params, output_params=output_params)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_b = rec.add(m, v)


class BornOperator(Operator):
    def __init__(self, dm, m, src, damp, rec, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)

        input_params = [dm, m, src, damp, rec]
        u = TimeData("u", m.shape, src.nt, time_order=time_order, save=False, dtype=m.dtype)
        U = TimeData("U", m.shape, src.nt, time_order=time_order, save=False, dtype=m.dtype)
        output_params = [u, U]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        dt = src.dt
        h = src.h
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[0]
        first_stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'), u, fw=True)
        second_stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'), U, fw=True)
        first_stencil_args = [m.indexed[space_dim], dt, h, damp.indexed[space_dim]]
        first_update = Eq(u.indexed[total_dim], u.indexed[total_dim]+first_stencil)
        src2 = (-(dt**-2)*(u.indexed[total_dim]-2*u.indexed[tuple((t - 1,) + space_dim)] +
                u.indexed[tuple((t - 2,) + space_dim)])*dm.indexed[space_dim])
        second_stencil_args = [m.indexed[space_dim], dt, h, damp.indexed[space_dim]]
        second_update = Eq(U.indexed[total_dim], second_stencil)
        insert_second_source = Eq(U.indexed[total_dim], U.indexed[total_dim]+(dt*dt)/m.indexed[space_dim]*src2)
        reset_u = Eq(u.indexed[tuple((t - 2,) + space_dim)], 0)
        stencils = [first_update, second_update, insert_second_source, reset_u]
        substitutions = [dict(zip(subs, first_stencil_args)),
                         dict(zip(subs, second_stencil_args)), {}, {}]

        super(BornOperator, self).__init__(src.nt, m.shape, stencils=stencils,
                                           substitutions=substitutions, spc_border=spc_order/2,
                                           time_order=time_order, forward=True, dtype=m.dtype,
                                           input_params=input_params, output_params=output_params)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_b = src.add(m, u)
        self.propagator.time_loop_stencils_a = rec.read(U)

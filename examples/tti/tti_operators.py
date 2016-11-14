import sympy
from sympy import *

from devito.dimension import x, y, z
from devito.finite_difference import centered, first_derivative, left, right
from devito.interfaces import DenseData, TimeData
from devito.operator import Operator
from examples.source_type import SourceLike


class ForwardOperator(Operator):
    def __init__(self, model, src, damp, data, time_order=2, spc_order=4, save=False,
                 trigonometry='normal', **kwargs):
        trigonometry = 'original'
        nrec, nt = data.shape

        dt = model.get_critical_dt()
        u = TimeData(name="u", shape=model.get_shape_comp(),
                     time_dim=nt, time_order=time_order,
                     space_order=spc_order,
                     save=save, dtype=damp.dtype)
        v = TimeData(name="v", shape=model.get_shape_comp(),
                     time_dim=nt, time_order=time_order,
                     space_order=spc_order,
                     save=save, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(),
                      dtype=damp.dtype)
        m.data[:] = model.padm()

        if model.epsilon is not None:
            epsilon = DenseData(name="epsilon", shape=model.get_shape_comp(),
                                dtype=damp.dtype)
            epsilon.data[:] = model.pad(model.epsilon)
        else:
            epsilon = 1.0

        if model.delta is not None:
            delta = DenseData(name="delta", shape=model.get_shape_comp(),
                              dtype=damp.dtype)
            delta.data[:] = model.pad(model.delta)
        else:
            delta = 1.0
        if model.theta is not None:
            theta = DenseData(name="theta", shape=model.get_shape_comp(),
                              dtype=damp.dtype)
            theta.data[:] = model.pad(model.theta)
        else:
            theta = 0

        if len(model.get_shape_comp()) == 3:
            if model.phi is not None:
                phi = DenseData(name="phi", shape=model.get_shape_comp(),
                                dtype=damp.dtype)
                phi.data[:] = model.pad(model.phi)
            else:
                phi = 0

        u.pad_time = save
        v.pad_time = save
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt,
                         h=model.get_spacing(),
                         coordinates=data.receiver_coords,
                         ndim=len(damp.shape),
                         dtype=damp.dtype,
                         nbpml=model.nbpml)

        def Bhaskarasin(angle):
            if angle == 0:
                return 0
            else:
                return (16.0 * angle * (3.1416 - abs(angle)) /
                        (49.3483 - 4.0 * abs(angle) * (3.1416 - abs(angle))))

        def Bhaskaracos(angle):
            if angle == 0:
                return 1.0
            else:
                return Bhaskarasin(angle + 1.5708)

        cos = Bhaskaracos if trigonometry == 'Bhaskara' else sympy.cos
        sin = Bhaskarasin if trigonometry == 'Bhaskara' else sympy.sin

        Hp, Hzr = symbols('Hp Hzr')
        if len(m.shape) == 3:
            ang0 = Function('ang0')(x, y, z)
            ang1 = Function('ang1')(x, y, z)
            ang2 = Function('ang2')(x, y, z)
            ang3 = Function('ang3')(x, y, z)
        else:
            ang0 = Function('ang0')(x, y)
            ang1 = Function('ang1')(x, y)

        s, h = symbols('s h')

        ang0 = cos(theta)
        ang1 = sin(theta)
        spc_brd = spc_order / 2
        # Derive stencil from symbolic equation
        if len(m.shape) == 3:
            ang2 = cos(phi)
            ang3 = sin(phi)

            Gy1p = (ang3 * u.dxl - ang2 * u.dyl)
            Gyy1 = (first_derivative(Gy1p, ang3, dim=x, side=right, order=spc_brd) -
                    first_derivative(Gy1p, ang2, dim=y, side=right, order=spc_brd))

            Gy2p = (ang3 * u.dxr - ang2 * u.dyr)
            Gyy2 = (first_derivative(Gy2p, ang3, dim=x, side=left, order=spc_brd) -
                    first_derivative(Gy2p, ang2, dim=y, side=left, order=spc_brd))

            Gx1p = (ang0 * ang2 * u.dxl + ang0 * ang3 * u.dyl - ang1 * u.dzl)
            Gz1r = (ang1 * ang2 * v.dxl + ang1 * ang3 * v.dyl + ang0 * v.dzl)
            Gxx1 = (first_derivative(Gx1p, ang0, ang2,
                                     dim=x, side=right, order=spc_brd) +
                    first_derivative(Gx1p, ang0, ang3,
                                     dim=y, side=right, order=spc_brd) -
                    first_derivative(Gx1p, ang1, dim=z, side=right, order=spc_brd))
            Gzz1 = (first_derivative(Gz1r, ang1, ang2,
                                     dim=x, side=right, order=spc_brd) +
                    first_derivative(Gz1r, ang1, ang3,
                                     dim=y, side=right, order=spc_brd) +
                    first_derivative(Gz1r, ang0, dim=z, side=right, order=spc_brd))

            Gx2p = (ang0 * ang2 * u.dxr + ang0 * ang3 * u.dyr - ang1 * u.dzr)
            Gz2r = (ang1 * ang2 * v.dxr + ang1 * ang3 * v.dyr + ang0 * v.dzr)
            Gxx2 = (first_derivative(Gx2p, ang0, ang2,
                                     dim=x, side=left, order=spc_brd) +
                    first_derivative(Gx2p, ang0, ang3,
                                     dim=y, side=left, order=spc_brd) -
                    first_derivative(Gx2p, ang1, dim=z, side=left, order=spc_brd))
            Gzz2 = (first_derivative(Gz2r, ang1, ang2,
                                     dim=x, side=left, order=spc_brd) +
                    first_derivative(Gz2r, ang1, ang3,
                                     dim=y, side=left, order=spc_brd) +
                    first_derivative(Gz2r, ang0,
                                     dim=z, side=left, order=spc_brd))
            parm = [m, damp, epsilon, delta, theta, phi, u, v]
        else:
            Gyy2 = 0
            Gyy1 = 0
            parm = [m, damp, epsilon, delta, theta, u, v]
            Gx1p = (ang0 * u.dxr - ang1 * u.dy)
            Gz1r = (ang1 * v.dxr + ang0 * v.dy)
            Gxx1 = (first_derivative(Gx1p * ang0, dim=x,
                                     side=left, order=spc_brd) -
                    first_derivative(Gx1p * ang1, dim=y,
                                     side=centered, order=spc_brd))
            Gzz1 = (first_derivative(Gz1r * ang1, dim=x,
                                     side=left, order=spc_brd) +
                    first_derivative(Gz1r * ang0, dim=y,
                                     side=centered, order=spc_brd))
            Gx2p = (ang0 * u.dx - ang1 * u.dyr)
            Gz2r = (ang1 * v.dx + ang0 * v.dyr)
            Gxx2 = (first_derivative(Gx2p * ang0, dim=x,
                                     side=centered, order=spc_brd) -
                    first_derivative(Gx2p * ang1, dim=y,
                                     side=left, order=spc_brd))
            Gzz2 = (first_derivative(Gz2r * ang1, dim=x,
                                     side=centered, order=spc_brd) +
                    first_derivative(Gz2r * ang0, dim=y,
                                     side=left, order=spc_brd))

        stencilp = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * u + (s * damp - 2.0 * m) *
             u.backward + 2.0 * s**2 * (epsilon * Hp + delta * Hzr))
        stencilr = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * v + (s * damp - 2.0 * m) *
             v.backward + 2.0 * s**2 * (delta * Hp + Hzr))

        Hp_val = -(.5 * Gxx1 + .5 * Gxx2 + .5 * Gyy1 + .5 * Gyy2)
        Hzr_val = -(.5 * Gzz1 + .5 * Gzz2)
        factorized = {Hp: Hp_val, Hzr: Hzr_val}
        # Add substitutions for spacing (temporal and spatial)
        subs = [{s: src.dt, h: src.h}, {s: src.dt, h: src.h}]
        first_stencil = Eq(u.forward, stencilp.xreplace(factorized))
        second_stencil = Eq(v.forward, stencilr.xreplace(factorized))
        stencils = [first_stencil, second_stencil]
        super(ForwardOperator, self).__init__(src.nt, m.shape,
                                              stencils=stencils,
                                              subs=subs,
                                              spc_border=spc_order/2 + 2,
                                              time_order=time_order,
                                              forward=True,
                                              dtype=m.dtype,
                                              input_params=parm,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [src, src.coordinates, rec, rec.coordinates]
        self.output_params += [v, rec]
        self.propagator.time_loop_stencils_a = (src.add(m, u) + src.add(m, v) +
                                                rec.read2(u, v))
        self.propagator.add_devito_param(src)
        self.propagator.add_devito_param(src.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class AdjointOperator(Operator):
    def __init__(self, m, rec, damp, srca, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)

        input_params = [m, rec, damp, srca]
        v = TimeData("v", m.shape, rec.nt, time_order=time_order,
                     save=True, dtype=m.dtype)
        output_params = [v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v.indexed[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, time_order, stencil,
                                           Function('p'), v, fw=False)
        main_stencil = Eq(lhs, stencil)
        stencil_args = [m.indexed[space_dim], rec.dt, rec.h,
                        damp.indexed[space_dim]]
        stencils = [main_stencil]
        substitutions = [dict(zip(subs, stencil_args))]

        super(AdjointOperator, self).__init__(rec.nt, m.shape,
                                              stencils=stencils,
                                              subs=substitutions,
                                              spc_border=spc_order/2,
                                              time_order=time_order,
                                              forward=False, dtype=m.dtype,
                                              input_params=input_params,
                                              output_params=output_params)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_a = rec.add(m, v) + srca.read(v)


class GradientOperator(Operator):
    def __init__(self, u, m, rec, damp, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)

        input_params = [u, m, rec, damp]
        v = TimeData("v", m.shape, rec.nt, time_order=time_order,
                     save=False, dtype=m.dtype)
        grad = DenseData("grad", m.shape, dtype=m.dtype)
        output_params = [grad, v]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        lhs = v.indexed[total_dim]
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[1]
        stencil = self.smart_sympy_replace(dim, time_order, stencil,
                                           Function('p'), v, fw=False)
        stencil_args = [m.indexed[space_dim], rec.dt, rec.h,
                        damp.indexed[space_dim]]
        main_stencil = Eq(lhs, lhs + stencil)
        gradient_update = Eq(grad.indexed[space_dim],
                             grad.indexed[space_dim] -
                             (v.indexed[total_dim] - 2 *
                              v.indexed[tuple((t + 1,) + space_dim)] +
                              v.indexed[tuple((t + 2,) + space_dim)]) *
                             u.indexed[total_dim])
        reset_v = Eq(v.indexed[tuple((t + 2,) + space_dim)], 0)
        stencils = [main_stencil, gradient_update, reset_v]
        substitutions = [dict(zip(subs, stencil_args)), {}, {}]

        super(GradientOperator, self).__init__(rec.nt, m.shape,
                                               stencils=stencils,
                                               subs=substitutions,
                                               spc_border=spc_order/2,
                                               time_order=time_order,
                                               forward=False, dtype=m.dtype,
                                               input_params=input_params,
                                               output_params=output_params)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_b = rec.add(m, v)


class BornOperator(Operator):
    def __init__(self, dm, m, src, damp, rec, time_order=4, spc_order=12):
        assert(m.shape == damp.shape)

        input_params = [dm, m, src, damp, rec]
        u = TimeData("u", m.shape, src.nt, time_order=time_order,
                     save=False, dtype=m.dtype)
        U = TimeData("U", m.shape, src.nt, time_order=time_order,
                     save=False, dtype=m.dtype)
        output_params = [u, U]
        dim = len(m.shape)
        total_dim = self.total_dim(dim)
        space_dim = self.space_dim(dim)
        dt = src.dt
        h = src.h
        stencil, subs = self._init_taylor(dim, time_order, spc_order)[0]
        first_stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'),
                                                 u, fw=True)
        second_stencil = self.smart_sympy_replace(dim, time_order, stencil, Function('p'),
                                                  U, fw=True)
        first_stencil_args = [m.indexed[space_dim], dt, h, damp.indexed[space_dim]]
        first_update = Eq(u.indexed[total_dim], u.indexed[total_dim]+first_stencil)
        src2 = (-(dt**-2)*(u.indexed[total_dim]-2*u.indexed[tuple((t - 1,) + space_dim)] +
                u.indexed[tuple((t - 2,) + space_dim)])*dm.indexed[space_dim])
        second_stencil_args = [m.indexed[space_dim], dt, h, damp.indexed[space_dim]]
        second_update = Eq(U.indexed[total_dim], second_stencil)
        insert_second_source = Eq(U.indexed[total_dim], U.indexed[total_dim] +
                                  (dt*dt)/m.indexed[space_dim]*src2)
        reset_u = Eq(u.indexed[tuple((t - 2,) + space_dim)], 0)
        stencils = [first_update, second_update, insert_second_source, reset_u]
        substitutions = [dict(zip(subs, first_stencil_args)),
                         dict(zip(subs, second_stencil_args)), {}, {}]

        super(BornOperator, self).__init__(src.nt, m.shape,
                                           stencils=stencils,
                                           subs=substitutions,
                                           spc_border=spc_order/2,
                                           time_order=time_order,
                                           forward=True,
                                           dtype=m.dtype,
                                           input_params=input_params,
                                           output_params=output_params)

        # Insert source and receiver terms post-hoc
        self.propagator.time_loop_stencils_b = src.add(m, u)
        self.propagator.time_loop_stencils_a = rec.read(U)

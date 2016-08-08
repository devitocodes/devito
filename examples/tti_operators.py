from sympy import Eq, Function
from sympy.abc import *

from devito.finite_difference import first_derivative
from devito.operator import *


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

        stencilp = 1.0 / (2.0 * m + s * damp) * (4.0 * m * u + (s * damp - 2.0 * m) * u.backward + 2.0 * s**2 * (A * Hp + B * Hzr))
        stencilr = 1.0 / (2.0 * m + s * damp) * (4.0 * m * v + (s * damp - 2.0 * m) * v.backward + 2.0 * s**2 * (B * Hp + Hzr))
        Hp = -(.5 * Gxx1 + .5 * Gxx2 + .5 * Gyy1 + .5 * Gyy2)
        Hzr = -(.5 * Gzz1 + .5 * Gzz2)
        factorized = {"Hp": Hp, "Hzr": Hzr}
        # Add substitutions for spacing (temporal and spatial)
        subs = [{s: src.dt, h: src.h}, {s: src.dt, h: src.h}]
        first_stencil = Eq(u.forward, stencilp)
        second_stencil = Eq(v.forward, stencilr)
        stencils = [first_stencil, second_stencil]
        super(ForwardOperator, self).__init__(src.nt, m.shape, stencils=stencils, substitutions=subs,
                                              spc_border=spc_order/2, time_order=time_order, forward=True, dtype=m.dtype,
                                              input_params=parm, factorized=factorized, **kwargs)
        # Insert source and receiver terms post-hoc
        self.input_params += [src, src.coordinates, rec, rec.coordinates]
        self.propagator.time_loop_stencils_a = src.add(m, u) + src.add(m, v) + rec.read2(u, v)
        self.output_params = [src]
        self.propagator.add_devito_param(src)
        self.propagator.add_devito_param(src.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
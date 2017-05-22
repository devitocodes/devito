from sympy import Eq, sin, cos
from sympy.abc import h, s

from devito import Operator, DenseData, x, y, z
from devito.finite_difference import centered, first_derivative, right, transpose


def ForwardOperator(model, u, v, src, rec, data, time_order=2,
                    spc_order=4, save=False, u_ini=None, **kwargs):
    """
    Constructor method for the forward modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    :param: u_ini : wavefield at the three first time step for non-zero initial condition
    """
    nt, nrec = data.shape
    nt, nsrc = src.shape
    dt = model.critical_dt
    m, damp = model.m, model.damp
    epsilon, delta, theta, phi = model.epsilon, model.delta, model.theta, model.phi
    parm = [m, damp, u, v]
    parm += [p for p in [epsilon, delta, theta, phi] if isinstance(p, DenseData)]

    spc_brd = spc_order/2

    ang0 = cos(theta)
    ang1 = sin(theta)
    if len(model.shape) == 3:
        ang2 = cos(phi)
        ang3 = sin(phi)

        # Derive stencil from symbolic equation
        Gyp = (ang3 * u.dx - ang2 * u.dyr)
        Gyy = (first_derivative(Gyp * ang3,
                                dim=x, side=centered, order=spc_brd,
                                matvec=transpose) -
               first_derivative(Gyp * ang2,
                                dim=y, side=right, order=spc_brd,
                                matvec=transpose))
        Gyp2 = (ang3 * u.dxr - ang2 * u.dy)
        Gyy2 = (first_derivative(Gyp2 * ang3,
                                 dim=x, side=right, order=spc_brd,
                                 matvec=transpose) -
                first_derivative(Gyp2 * ang2,
                                 dim=y, side=centered, order=spc_brd,
                                 matvec=transpose))

        Gxp = (ang0 * ang2 * u.dx + ang0 * ang3 * u.dyr - ang1 * u.dzr)
        Gzr = (ang1 * ang2 * v.dx + ang1 * ang3 * v.dyr + ang0 * v.dzr)
        Gxx = (first_derivative(Gxp * ang0 * ang2,
                                dim=x, side=centered, order=spc_brd,
                                matvec=transpose) +
               first_derivative(Gxp * ang0 * ang3,
                                dim=y, side=right, order=spc_brd,
                                matvec=transpose) -
               first_derivative(Gxp * ang1,
                                dim=z, side=right, order=spc_brd,
                                matvec=transpose))
        Gzz = (first_derivative(Gzr * ang1 * ang2,
                                dim=x, side=centered, order=spc_brd,
                                matvec=transpose) +
               first_derivative(Gzr * ang1 * ang3,
                                dim=y, side=right, order=spc_brd,
                                matvec=transpose) +
               first_derivative(Gzr * ang0,
                                dim=z, side=right, order=spc_brd,
                                matvec=transpose))
        Gxp2 = (ang0 * ang2 * u.dxr + ang0 * ang3 * u.dy - ang1 * u.dz)
        Gzr2 = (ang1 * ang2 * v.dxr + ang1 * ang3 * v.dy + ang0 * v.dz)
        Gxx2 = (first_derivative(Gxp2 * ang0 * ang2,
                                 dim=x, side=right, order=spc_brd,
                                 matvec=transpose) +
                first_derivative(Gxp2 * ang0 * ang3,
                                 dim=y, side=centered, order=spc_brd,
                                 matvec=transpose) -
                first_derivative(Gxp2 * ang1,
                                 dim=z, side=centered, order=spc_brd,
                                 matvec=transpose))
        Gzz2 = (first_derivative(Gzr2 * ang1 * ang2,
                                 dim=x, side=right, order=spc_brd,
                                 matvec=transpose) +
                first_derivative(Gzr2 * ang1 * ang3,
                                 dim=y, side=centered, order=spc_brd,
                                 matvec=transpose) +
                first_derivative(Gzr2 * ang0,
                                 dim=z, side=centered, order=spc_brd,
                                 matvec=transpose))
        Hp = -(.5*Gxx + .5*Gxx2 + .5 * Gyy + .5*Gyy2)
        Hzr = -(.5*Gzz + .5 * Gzz2)

    else:
        Gx1p = (ang0 * u.dxr - ang1 * u.dy)
        Gz1r = (ang1 * v.dxr + ang0 * v.dy)
        Gxx1 = (first_derivative(Gx1p * ang0, dim=x,
                                 side=right, order=spc_brd,
                                 matvec=transpose) -
                first_derivative(Gx1p * ang1, dim=y,
                                 side=centered, order=spc_brd,
                                 matvec=transpose))
        Gzz1 = (first_derivative(Gz1r * ang1, dim=x,
                                 side=right, order=spc_brd,
                                 matvec=transpose) +
                first_derivative(Gz1r * ang0, dim=y,
                                 side=centered, order=spc_brd,
                                 matvec=transpose))
        Gx2p = (ang0 * u.dx - ang1 * u.dyr)
        Gz2r = (ang1 * v.dx + ang0 * v.dyr)
        Gxx2 = (first_derivative(Gx2p * ang0, dim=x,
                                 side=centered, order=spc_brd,
                                 matvec=transpose) -
                first_derivative(Gx2p * ang1, dim=y,
                                 side=right, order=spc_brd,
                                 matvec=transpose))
        Gzz2 = (first_derivative(Gz2r * ang1, dim=x,
                                 side=centered, order=spc_brd,
                                 matvec=transpose) +
                first_derivative(Gz2r * ang0, dim=y,
                                 side=right, order=spc_brd,
                                 matvec=transpose))

        Hp = -(.5 * Gxx1 + .5 * Gxx2)
        Hzr = -(.5 * Gzz1 + .5 * Gzz2)

    stencilp = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * u + (s * damp - 2.0 * m) *
         u.backward + 2.0 * s**2 * (epsilon * Hp + delta * Hzr))
    stencilr = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * v + (s * damp - 2.0 * m) *
         v.backward + 2.0 * s**2 * (delta * Hp + Hzr))

    # Add substitutions for spacing (temporal and spatial)
    subs = {s: dt, h: model.get_spacing()}
    first_stencil = Eq(u.forward, stencilp)
    second_stencil = Eq(v.forward, stencilr)
    stencils = [first_stencil, second_stencil]

    dse = kwargs.get('dse', 'advanced')
    dle = kwargs.get('dle', 'advanced')
    ti = u.indices[0]
    stencils += src.inject(field=u, u_t=ti + 1, expr=src * dt * dt / m,
                           offset=model.nbpml)
    stencils += src.inject(field=v, u_t=ti + 1, expr=src * dt * dt / m,
                           offset=model.nbpml)
    stencils += rec.interpolate(expr=u, u_t=ti, offset=model.nbpml)
    stencils += rec.interpolate(expr=v, u_t=ti, offset=model.nbpml)

    return Operator(stencils, subs=subs, dse=dse, dle=dle)

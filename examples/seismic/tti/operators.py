from sympy import Eq, cos, sin

from devito import Operator, TimeData
from examples.seismic import PointSource, Receiver
from devito.finite_difference import centered, first_derivative, right, transpose
from devito.dimension import x, y, z, t, time


def ForwardOperator(model, source, receiver, time_order=2, space_order=4,
                    save=False, **kwargs):
    """
    Constructor method for the forward modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    :param: u_ini : wavefield at the three first time step for non-zero initial condition
    """
    dt = model.critical_dt

    m, damp, epsilon, delta, theta, phi = (model.m, model.damp, model.epsilon,
                                           model.delta, model.theta, model.phi)

    # Create symbols for forward wavefield, source and receivers
    u = TimeData(name='u', shape=model.shape_domain, time_dim=source.nt,
                 time_order=time_order, space_order=space_order, save=save,
                 dtype=model.dtype)
    v = TimeData(name='v', shape=model.shape_domain, time_dim=source.nt,
                 time_order=time_order, space_order=space_order, save=save,
                 dtype=model.dtype)
    src = PointSource(name='src', ntime=source.nt, ndim=source.ndim,
                      npoint=source.npoint)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    ang0 = cos(theta)
    ang1 = sin(theta)
    if len(model.shape) == 3:
        ang2 = cos(phi)
        ang3 = sin(phi)

        # Derive stencil from symbolic equation
        Gyp = (ang3 * u.dx - ang2 * u.dyr)
        Gyy = (first_derivative(Gyp * ang3,
                                dim=x, side=centered, order=space_order,
                                matvec=transpose) -
               first_derivative(Gyp * ang2,
                                dim=y, side=right, order=space_order,
                                matvec=transpose))
        Gyp2 = (ang3 * u.dxr - ang2 * u.dy)
        Gyy2 = (first_derivative(Gyp2 * ang3,
                                 dim=x, side=right, order=space_order,
                                 matvec=transpose) -
                first_derivative(Gyp2 * ang2,
                                 dim=y, side=centered, order=space_order,
                                 matvec=transpose))

        Gxp = (ang0 * ang2 * u.dx + ang0 * ang3 * u.dyr - ang1 * u.dzr)
        Gzr = (ang1 * ang2 * v.dx + ang1 * ang3 * v.dyr + ang0 * v.dzr)
        Gxx = (first_derivative(Gxp * ang0 * ang2,
                                dim=x, side=centered, order=space_order,
                                matvec=transpose) +
               first_derivative(Gxp * ang0 * ang3,
                                dim=y, side=right, order=space_order,
                                matvec=transpose) -
               first_derivative(Gxp * ang1,
                                dim=z, side=right, order=space_order,
                                matvec=transpose))
        Gzz = (first_derivative(Gzr * ang1 * ang2,
                                dim=x, side=centered, order=space_order,
                                matvec=transpose) +
               first_derivative(Gzr * ang1 * ang3,
                                dim=y, side=right, order=space_order,
                                matvec=transpose) +
               first_derivative(Gzr * ang0,
                                dim=z, side=right, order=space_order,
                                matvec=transpose))
        Gxp2 = (ang0 * ang2 * u.dxr + ang0 * ang3 * u.dy - ang1 * u.dz)
        Gzr2 = (ang1 * ang2 * v.dxr + ang1 * ang3 * v.dy + ang0 * v.dz)
        Gxx2 = (first_derivative(Gxp2 * ang0 * ang2,
                                 dim=x, side=right, order=space_order,
                                 matvec=transpose) +
                first_derivative(Gxp2 * ang0 * ang3,
                                 dim=y, side=centered, order=space_order,
                                 matvec=transpose) -
                first_derivative(Gxp2 * ang1,
                                 dim=z, side=centered, order=space_order,
                                 matvec=transpose))
        Gzz2 = (first_derivative(Gzr2 * ang1 * ang2,
                                 dim=x, side=right, order=space_order,
                                 matvec=transpose) +
                first_derivative(Gzr2 * ang1 * ang3,
                                 dim=y, side=centered, order=space_order,
                                 matvec=transpose) +
                first_derivative(Gzr2 * ang0,
                                 dim=z, side=centered, order=space_order,
                                 matvec=transpose))
        Hp = -(.5*Gxx + .5*Gxx2 + .5 * Gyy + .5*Gyy2)
        Hzr = -(.5*Gzz + .5 * Gzz2)

    else:
        Gx1p = (ang0 * u.dxr - ang1 * u.dy)
        Gz1r = (ang1 * v.dxr + ang0 * v.dy)
        Gxx1 = (first_derivative(Gx1p * ang0, dim=x,
                                 side=right, order=space_order,
                                 matvec=transpose) -
                first_derivative(Gx1p * ang1, dim=y,
                                 side=centered, order=space_order,
                                 matvec=transpose))
        Gzz1 = (first_derivative(Gz1r * ang1, dim=x,
                                 side=right, order=space_order,
                                 matvec=transpose) +
                first_derivative(Gz1r * ang0, dim=y,
                                 side=centered, order=space_order,
                                 matvec=transpose))
        Gx2p = (ang0 * u.dx - ang1 * u.dyr)
        Gz2r = (ang1 * v.dx + ang0 * v.dyr)
        Gxx2 = (first_derivative(Gx2p * ang0, dim=x,
                                 side=centered, order=space_order,
                                 matvec=transpose) -
                first_derivative(Gx2p * ang1, dim=y,
                                 side=right, order=space_order,
                                 matvec=transpose))
        Gzz2 = (first_derivative(Gz2r * ang1, dim=x,
                                 side=centered, order=space_order,
                                 matvec=transpose) +
                first_derivative(Gz2r * ang0, dim=y,
                                 side=right, order=space_order,
                                 matvec=transpose))

        Hp = -(.5 * Gxx1 + .5 * Gxx2)
        Hzr = -(.5 * Gzz1 + .5 * Gzz2)

    s = t.spacing

    stencilp = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * u + (s * damp - 2.0 * m) *
         u.backward + 2.0 * s**2 * (epsilon * Hp + delta * Hzr))
    stencilr = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * v + (s * damp - 2.0 * m) *
         v.backward + 2.0 * s**2 * (delta * Hp + Hzr))

    # Add substitutions for spacing (temporal and spatial)
    first_stencil = Eq(u.forward, stencilp)
    second_stencil = Eq(v.forward, stencilr)
    stencils = [first_stencil, second_stencil]

    ti = u.indices[0]
    stencils += src.inject(field=u, u_t=ti + 1, expr=src * dt * dt / m,
                           offset=model.nbpml)
    stencils += src.inject(field=v, u_t=ti + 1, expr=src * dt * dt / m,
                           offset=model.nbpml)
    stencils += rec.interpolate(expr=u + v, u_t=ti, offset=model.nbpml)
    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.get_spacing()[j]) for i, j
                 in zip(u.indices[1:], range(len(model.shape)))])
    return Operator(stencils, subs=subs, name='ForwardTTI', **kwargs)

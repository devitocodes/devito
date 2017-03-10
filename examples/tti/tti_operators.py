from sympy import *

from devito.dimension import x, y, z, t, time
from devito.finite_difference import centered, first_derivative, right, transpose
from devito.interfaces import DenseData
from devito.operator import Operator
from devito.stencilkernel import StencilKernel


def ForwardOperator(model, u, v, src, rec, damp, data, time_order=2,
                    spc_order=4, save=False, u_ini=None, legacy=True,
                    **kwargs):
    """
    Constructor method for the forward modelling operator in an acoustic media

    :param model: IGrid() object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param damp: Dampening coeeficents for the ABCs
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    :param: u_ini : wavefield at the three first time step for non-zero initial condition
    """
    nt, nrec = data.shape
    nt, nsrc = src.shape
    dt = model.get_critical_dt()

    m = DenseData(name="m", shape=model.get_shape_comp(),
                  dtype=damp.dtype, space_order=spc_order)
    m.data[:] = model.padm()

    parm = [m, damp, u, v]

    if model.epsilon is not None:
        epsilon = DenseData(name="epsilon", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
        epsilon.data[:] = model.pad(model.epsilon)
        parm += [epsilon]
    else:
        epsilon = 1

    if model.delta is not None:
        delta = DenseData(name="delta", shape=model.get_shape_comp(),
                          dtype=damp.dtype, space_order=spc_order)
        delta.data[:] = model.pad(model.delta)
        parm += [delta]
    else:
        delta = 1

    if model.theta is not None:
        theta = DenseData(name="theta", shape=model.get_shape_comp(),
                          dtype=damp.dtype, space_order=spc_order)
        theta.data[:] = model.pad(model.theta)
        parm += [theta]
    else:
        theta = 0

    if model.phi is not None:
        phi = DenseData(name="phi", shape=model.get_shape_comp(),
                        dtype=damp.dtype, space_order=spc_order)
        phi.data[:] = model.pad(model.phi)
        parm += [phi]
    else:
        phi = 0

    s, h = symbols('s h')
    P, R = symbols('P R')
    spc_brd = spc_order/2

    ang0 = cos(theta)
    ang1 = sin(theta)
    if len(m.shape) == 3:
        ang2 = cos(phi)
        ang3 = sin(phi)

        # Derive stencil from symbolic equation
        Gyp = (ang3 * u.dx - ang2 * u.dyr)
        Gyy = (first_derivative(Gyp * ang3,
                                dim=x, side=centered,
                                order=spc_brd, matvec=transpose) -
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
        Hp = -(.5*Gxx + .5*Gxx2 + .5*Gyy + .5*Gyy2)
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

    eqp = m * u.dt2 - epsilon * P - delta * R + damp * u.dt
    eqr = m * v.dt2 - delta * P - R + damp * v.dt
    stencilp = solve(eqp, u.forward)[0]
    stencilr = solve(eqr, v.forward)[0]
    # Add substitutions for spacing (temporal and spatial)
    subs = {s: dt, h: model.get_spacing()}
    first_stencil = Eq(u.forward, stencilp.xreplace({P: Hp, R: Hzr}))
    second_stencil = Eq(v.forward, stencilr.xreplace({P: Hp, R: Hzr}))
    stencils = [first_stencil, second_stencil]

    if legacy:
        kwargs.pop('dle', None)
        kwargs.pop('dse', None)
        op = Operator(nt, m.shape, stencils=stencils, subs=[subs, subs],
                      spc_border=spc_order, time_order=time_order,
                      forward=True, dtype=m.dtype, input_params=parm,
                      **kwargs)

        # Insert source and receiver terms post-hoc
        op.input_params += [src, src.coordinates, rec, rec.coordinates]
        op.output_params += [v, rec]
        op.propagator.time_loop_stencils_a = (src.add(m, u) + src.add(m, v) +
                                              rec.read2(u, v))
        op.propagator.add_devito_param(src)
        op.propagator.add_devito_param(src.coordinates)
        op.propagator.add_devito_param(rec)
        op.propagator.add_devito_param(rec.coordinates)

    else:
        dse = kwargs.get('dse', 'advanced')
        dle = kwargs.get('dle', 'advanced')
        compiler = kwargs.get('compiler', None)

        stencils += src.point2grid(u, m, u_t=t, p_t=time)
        stencils += src.point2grid(v, m, u_t=t, p_t=time)
        stencils += rec.grid2point(u) + rec.grid2point(v)

        op = StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                           compiler=compiler)

    return op

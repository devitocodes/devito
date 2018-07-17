from sympy import cos, sin

from devito import Eq, Operator, TimeFunction
from examples.seismic import PointSource, Receiver
from devito.finite_difference import (centered, first_derivative, right, transpose, left,
                                      staggered_diff)


def second_order_stencil(model, u, v, H0, Hz):
    """
    Creates the stencil corresponding to the second order TTI wave equation
    u.dt2 =  (epsilon * H0 + delta * Hz) - damp * u.dt
    v.dt2 =  (delta * H0 + Hz) - damp * v.dt
    """
    # Stencils
    m, damp, delta, epsilon = model.m, model.damp, model.delta, model.epsilon
    s = model.grid.stepping_dim.spacing
    stencilp = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * u + (s * damp - 2.0 * m) *
         u.backward + 2.0 * s ** 2 * (epsilon * H0 + delta * Hz))
    stencilr = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * v + (s * damp - 2.0 * m) *
         v.backward + 2.0 * s ** 2 * (delta * H0 + Hz))
    first_stencil = Eq(u.forward, stencilp)
    second_stencil = Eq(v.forward, stencilr)
    stencils = [first_stencil, second_stencil]
    return stencils


def Gxx_shifted(field, costheta, sintheta, cosphi, sinphi, space_order):
    """
    3D rotated second order derivative in the direction x as an average of
    two non-centered rotated second order derivative in the direction x
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt x
    """
    x, y, z = field.space_dimensions
    Gx1 = (costheta * cosphi * field.dx + costheta * sinphi * field.dyr -
           sintheta * field.dzr)
    Gxx1 = (first_derivative(Gx1 * costheta * cosphi,
                             dim=x, side=centered, order=space_order,
                             matvec=transpose) +
            first_derivative(Gx1 * costheta * sinphi,
                             dim=y, side=right, order=space_order,
                             matvec=transpose) -
            first_derivative(Gx1 * sintheta,
                             dim=z, side=right, order=space_order,
                             matvec=transpose))
    Gx2 = (costheta * cosphi * field.dxr + costheta * sinphi * field.dy -
           sintheta * field.dz)
    Gxx2 = (first_derivative(Gx2 * costheta * cosphi,
                             dim=x, side=right, order=space_order,
                             matvec=transpose) +
            first_derivative(Gx2 * costheta * sinphi,
                             dim=y, side=centered, order=space_order,
                             matvec=transpose) -
            first_derivative(Gx2 * sintheta,
                             dim=z, side=centered, order=space_order,
                             matvec=transpose))
    return -.5 * (Gxx1 + Gxx2)


def Gxx_shifted_2d(field, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction x as an average of
    two non-centered rotated second order derivative in the direction x
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt x
    """
    x, y = field.space_dimensions[:2]
    Gx1 = (costheta * field.dxr - sintheta * field.dy)
    Gxx1 = (first_derivative(Gx1 * costheta, dim=x,
                             side=right, order=space_order,
                             matvec=transpose) -
            first_derivative(Gx1 * sintheta, dim=y,
                             side=centered, order=space_order,
                             matvec=transpose))
    Gx2p = (costheta * field.dx - sintheta * field.dyr)
    Gxx2 = (first_derivative(Gx2p * costheta, dim=x,
                             side=centered, order=space_order,
                             matvec=transpose) -
            first_derivative(Gx2p * sintheta, dim=y,
                             side=right, order=space_order,
                             matvec=transpose))

    return -.5 * (Gxx1 + Gxx2)


def Gyy_shifted(field, cosphi, sinphi, space_order):
    """
    3D rotated second order derivative in the direction y as an average of
    two non-centered rotated second order derivative in the direction y
    :param field: symbolic data whose derivative we are computing
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt y
    """
    x, y = field.space_dimensions[:2]
    Gyp = (sinphi * field.dx - cosphi * field.dyr)
    Gyy = (first_derivative(Gyp * sinphi,
                            dim=x, side=centered, order=space_order,
                            matvec=transpose) -
           first_derivative(Gyp * cosphi,
                            dim=y, side=right, order=space_order,
                            matvec=transpose))
    Gyp2 = (sinphi * field.dxr - cosphi * field.dy)
    Gyy2 = (first_derivative(Gyp2 * sinphi,
                             dim=x, side=right, order=space_order,
                             matvec=transpose) -
            first_derivative(Gyp2 * cosphi,
                             dim=y, side=centered, order=space_order,
                             matvec=transpose))
    return -.5 * (Gyy + Gyy2)


def Gzz_shifted(field, costheta, sintheta, cosphi, sinphi, space_order):
    """
    3D rotated second order derivative in the direction z as an average of
    two non-centered rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt z
    """
    x, y, z = field.space_dimensions
    Gzr = (sintheta * cosphi * field.dx + sintheta * sinphi * field.dyr +
           costheta * field.dzr)
    Gzz = (first_derivative(Gzr * sintheta * cosphi,
                            dim=x, side=centered, order=space_order,
                            matvec=transpose) +
           first_derivative(Gzr * sintheta * sinphi,
                            dim=y, side=right, order=space_order,
                            matvec=transpose) +
           first_derivative(Gzr * costheta,
                            dim=z, side=right, order=space_order,
                            matvec=transpose))
    Gzr2 = (sintheta * cosphi * field.dxr + sintheta * sinphi * field.dy +
            costheta * field.dz)
    Gzz2 = (first_derivative(Gzr2 * sintheta * cosphi,
                             dim=x, side=right, order=space_order,
                             matvec=transpose) +
            first_derivative(Gzr2 * sintheta * sinphi,
                             dim=y, side=centered, order=space_order,
                             matvec=transpose) +
            first_derivative(Gzr2 * costheta,
                             dim=z, side=centered, order=space_order,
                             matvec=transpose))
    return -.5 * (Gzz + Gzz2)


def Gzz_shifted_2d(field, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction z as an average of
    two non-centered rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt
    :param sintheta:  sine of the tilt
    :param space_order: discretization order
    :return: rotated second order derivative wrt z
    """
    x, y = field.space_dimensions[:2]
    Gz1r = (sintheta * field.dxr + costheta * field.dy)
    Gzz1 = (first_derivative(Gz1r * sintheta, dim=x,
                             side=right, order=space_order,
                             matvec=transpose) +
            first_derivative(Gz1r * costheta, dim=y,
                             side=centered, order=space_order,
                             matvec=transpose))
    Gz2r = (sintheta * field.dx + costheta * field.dyr)
    Gzz2 = (first_derivative(Gz2r * sintheta, dim=x,
                             side=centered, order=space_order,
                             matvec=transpose) +
            first_derivative(Gz2r * costheta, dim=y,
                             side=right, order=space_order,
                             matvec=transpose))

    return -.5 * (Gzz1 + Gzz2)


def Gzz_centered(field, costheta, sintheta, cosphi, sinphi, space_order):
    """
    3D rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt z
    """
    order1 = space_order / 2
    x, y, z = field.space_dimensions
    Gz = -(sintheta * cosphi * first_derivative(field, dim=x,
                                                side=centered, order=order1) +
           sintheta * sinphi * first_derivative(field, dim=y,
                                                side=centered, order=order1) +
           costheta * first_derivative(field, dim=z,
                                       side=centered, order=order1))
    Gzz = (first_derivative(Gz * sintheta * cosphi,
                            dim=x, side=centered, order=order1,
                            matvec=transpose) +
           first_derivative(Gz * sintheta * sinphi,
                            dim=y, side=centered, order=order1,
                            matvec=transpose) +
           first_derivative(Gz * costheta,
                            dim=z, side=centered, order=order1,
                            matvec=transpose))
    return Gzz


def Gzz_centered_2d(field, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt z
    """
    order1 = space_order / 2
    x, y = field.space_dimensions[:2]
    Gz = -(sintheta * first_derivative(field, dim=x, side=centered, order=order1) +
           costheta * first_derivative(field, dim=y, side=centered, order=order1))
    Gzz = (first_derivative(Gz * sintheta, dim=x,
                            side=centered, order=order1,
                            matvec=transpose) +
           first_derivative(Gz * costheta, dim=y,
                            side=centered, order=order1,
                            matvec=transpose))
    return Gzz


# Centered case produces directly Gxx + Gyy
def Gxxyy_centered(field, costheta, sintheta, cosphi, sinphi, space_order):
    """
    Sum of the 3D rotated second order derivative in the direction x and y.
    As the Laplacian is rotation invariant, it is computed as the conventional
    Laplacian minus the second order rotated second order derivative in the direction z
    Gxx + Gyy = field.laplace - Gzz
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: Sum of the 3D rotated second order derivative in the direction x and y
    """
    Gzz = Gzz_centered(field, costheta, sintheta, cosphi, sinphi, space_order)
    return field.laplace - Gzz


def Gxx_centered_2d(field, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction x.
    As the Laplacian is rotation invariant, it is computed as the conventional
    Laplacian minus the second order rotated second order derivative in the direction z
    Gxx = field.laplace - Gzz
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: Sum of the 3D rotated second order derivative in the direction x
    """
    return field.laplace - Gzz_centered_2d(field, costheta, sintheta, space_order)


def kernel_shifted_2d(model, u, v, space_order):
    """
    TTI finite difference kernel. The equation we solve is:

    u.dt2 = (1+2 *epsilon) (Gxx(u)) + sqrt(1+ 2*delta) Gzz(v)
    v.dt2 = sqrt(1+ 2*delta) (Gxx(u)) +  Gzz(v)

    where epsilon and delta are the thomsen parameters. This function computes
    H0 = Gxx(u) + Gyy(u)
    Hz = Gzz(v)

    :param u: first TTI field
    :param v: second TTI field
    :param space_order: discretization order
    :return: u and v component of the rotated Laplacian in 2D
    """
    # Tilt and azymuth setup
    costheta = cos(model.theta)
    sintheta = sin(model.theta)

    Gxx = Gxx_shifted_2d(u, costheta, sintheta, space_order)
    Gzz = Gzz_shifted_2d(v, costheta, sintheta, space_order)
    return second_order_stencil(model, u, v, Gxx, Gzz)


def kernel_shifted_3d(model, u, v, space_order):
    """
    TTI finite difference kernel. The equation we solve is:

    u.dt2 = (1+2 *epsilon) (Gxx(u)+Gyy(u)) + sqrt(1+ 2*delta) Gzz(v)
    v.dt2 = sqrt(1+ 2*delta) (Gxx(u)+Gyy(u)) +  Gzz(v)

    where epsilon and delta are the thomsen parameters. This function computes
    H0 = Gxx(u) + Gyy(u)
    Hz = Gzz(v)

    :param u: first TTI field
    :param v: second TTI field
    :param space_order: discretization order
    :return: u and v component of the rotated Laplacian in 3D
    """
    # Tilt and azymuth setup
    costheta = cos(model.theta)
    sintheta = sin(model.theta)
    cosphi = cos(model.phi)
    sinphi = sin(model.phi)

    Gxx = Gxx_shifted(u, costheta, sintheta, cosphi, sinphi, space_order)
    Gyy = Gyy_shifted(u, cosphi, sinphi, space_order)
    Gzz = Gzz_shifted(v, costheta, sintheta, cosphi, sinphi, space_order)
    return second_order_stencil(model, u, v, Gxx + Gyy, Gzz)


def kernel_centered_2d(model, u, v, space_order):
    """
    TTI finite difference kernel. The equation we solve is:

    u.dt2 = (1+2 *epsilon) (Gxx(u)) + sqrt(1+ 2*delta) Gzz(v)
    v.dt2 = sqrt(1+ 2*delta) (Gxx(u)) +  Gzz(v)

    where epsilon and delta are the thomsen parameters. This function computes
    H0 = Gxx(u) + Gyy(u)
    Hz = Gzz(v)

    :param u: first TTI field
    :param v: second TTI field
    :param space_order: discretization order
    :return: u and v component of the rotated Laplacian in 2D
    """
    # Tilt and azymuth setup
    costheta = cos(model.theta)
    sintheta = sin(model.theta)

    Gxx = Gxx_centered_2d(u, costheta, sintheta, space_order)
    Gzz = Gzz_centered_2d(v, costheta, sintheta, space_order)
    return second_order_stencil(model, u, v, Gxx, Gzz)


def kernel_centered_3d(model, u, v, space_order):
    """
    TTI finite difference kernel. The equation we solve is:

    u.dt2 = (1+2 *epsilon) (Gxx(u)+Gyy(u)) + sqrt(1+ 2*delta) Gzz(v)
    v.dt2 = sqrt(1+ 2*delta) (Gxx(u)+Gyy(u)) +  Gzz(v)

    where epsilon and delta are the thomsen parameters. This function computes
    H0 = Gxx(u) + Gyy(u)
    Hz = Gzz(v)

    :param u: first TTI field
    :param v: second TTI field
    :return: u and v component of the rotated Laplacian in 2D
    """
    # Tilt and azymuth setup
    costheta = cos(model.theta)
    sintheta = sin(model.theta)
    cosphi = cos(model.phi)
    sinphi = sin(model.phi)

    Gxx = Gxxyy_centered(u, costheta, sintheta, cosphi, sinphi, space_order)
    Gzz = Gzz_centered(v, costheta, sintheta, cosphi, sinphi, space_order)
    return second_order_stencil(model, u, v, Gxx, Gzz)


def kernel_staggered_2d(model, u, v, space_order):
    """
    TTI finite difference. The equation solved is:
    vx.dt = - u.dx
    vz.dt = - v.dx
    m * v.dt = - sqrt(1 + 2 delta) vx.dx - vz.dz + Fh
    m * u.dt = - (1 + 2 epsilon) vx.dx - sqrt(1 + 2 delta) vz.dz + Fv
    """
    dampl = 1 - model.damp
    m, epsilon, delta, theta = (model.m, model.epsilon, model.delta, model.theta)
    s = model.grid.stepping_dim.spacing
    # Staggered setup
    stagg_x = (0, 1, 0)
    stagg_z = (0, 0, 1)
    x, z = model.grid.dimensions
    # Create symbols for forward wavefield, source and receivers
    vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                      time_order=1, space_order=space_order)
    vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                      time_order=1, space_order=space_order)

    # Stencils
    phdx = staggered_diff(u, dim=x, order=space_order, stagger=left, theta=theta)
    u_vx = Eq(vx.forward, dampl * vx - dampl * s * phdx)

    pvdz = staggered_diff(v, dim=z, order=space_order, stagger=left, theta=theta)
    u_vz = Eq(vz.forward, dampl * vz - dampl * s * pvdz)

    dvx = staggered_diff(vx.forward, dim=x, order=space_order, stagger=right,
                         theta=theta)
    dvz = staggered_diff(vz.forward, dim=z, order=space_order, stagger=right,
                         theta=theta)

    # u and v equations
    pv_eq = Eq(v.forward, dampl * (v - s / m * (delta * dvx + dvz)))

    ph_eq = Eq(u.forward, dampl * (u - s / m * (epsilon * dvx + delta * dvz)))

    return [u_vx, u_vz] + [pv_eq, ph_eq]


def kernel_staggered_3d(model, u, v, space_order):
    """
    TTI finite difference. The equation solved is:
    vx.dt = - u.dx
    vy.dt = - u.dx
    vz.dt = - v.dx
    m * v.dt = - sqrt(1 + 2 delta) (vx.dx + vy.dy) - vz.dz + Fh
    m * u.dt = - (1 + 2 epsilon) (vx.dx + vy.dy) - sqrt(1 + 2 delta) vz.dz + Fv
    """
    dampl = 1 - model.damp
    m, epsilon, delta, theta, phi = (model.m, model.epsilon, model.delta,
                                     model.theta, model.phi)
    s = model.grid.stepping_dim.spacing
    # Staggered setup
    stagg_x = (0, 1, 0, 0)
    stagg_y = (0, 0, 1, 0)
    stagg_z = (0, 0, 0, 1)
    x, y, z = model.grid.dimensions
    # Create symbols for forward wavefield, source and receivers
    vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                      time_order=1, space_order=space_order)
    vy = TimeFunction(name='vy', grid=model.grid, staggered=stagg_y,
                      time_order=1, space_order=space_order)
    vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                      time_order=1, space_order=space_order)

    # Stencils
    phdx = staggered_diff(u, dim=x, order=space_order, stagger=left,
                          theta=theta, phi=phi)
    u_vx = Eq(vx.forward, dampl * vx - dampl * s * phdx)

    phdy = staggered_diff(u, dim=y, order=space_order, stagger=left,
                          theta=theta, phi=phi)
    u_vy = Eq(vy.forward, dampl * vy - dampl * s * phdy)

    pvdz = staggered_diff(v, dim=z, order=space_order, stagger=left,
                          theta=theta, phi=phi)
    u_vz = Eq(vz.forward, dampl * vz - dampl * s * pvdz)

    dvx = staggered_diff(vx.forward, dim=x, order=space_order, stagger=right,
                         theta=theta, phi=phi)
    dvy = staggered_diff(vy.forward, dim=y, order=space_order, stagger=right,
                         theta=theta, phi=phi)
    dvz = staggered_diff(vz.forward, dim=z, order=space_order, stagger=right,
                         theta=theta, phi=phi)

    # u and v equations
    pv_eq = Eq(v.forward, dampl * (v - s / m * (delta * (dvx + dvy) + dvz)))

    ph_eq = Eq(u.forward, dampl * (u - s / m * (epsilon * (dvx + dvy) + delta * dvz)))

    return [u_vx, u_vy, u_vz] + [pv_eq, ph_eq]


def ForwardOperator(model, source, receiver, space_order=4,
                    save=False, kernel='centered', **kwargs):
    """
       Constructor method for the forward modelling operator in an acoustic media

       :param model: :class:`Model` object containing the physical parameters
       :param src: None ot IShot() (not currently supported properly)
       :param data: IShot() object containing the acquisition geometry and field data
       :param: time_order: Time discretization order
       :param: spc_order: Space discretization order
       """

    dt = model.grid.time_dim.spacing
    m = model.m
    time_order = 1 if kernel == 'staggered' else 2
    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=source.nt if save else None,
                     time_order=time_order, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid,
                     save=source.nt if save else None,
                     time_order=time_order, space_order=space_order)
    src = PointSource(name='src', grid=model.grid, time_range=source.time_range,
                      npoint=source.npoint)
    rec = Receiver(name='rec', grid=model.grid, time_range=receiver.time_range,
                   npoint=receiver.npoint)

    # FD kernels of the PDE
    FD_kernel = kernels[(kernel, len(model.shape))]
    stencils = FD_kernel(model, u, v, space_order)

    # Source and receivers
    stencils += src.inject(field=u.forward, expr=src * dt**2 / m,
                           offset=model.nbpml)
    stencils += src.inject(field=v.forward, expr=src * dt**2 / m,
                           offset=model.nbpml)
    stencils += rec.interpolate(expr=u + v, offset=model.nbpml)

    # Substitute spacing terms to reduce flops
    return Operator(stencils, subs=model.spacing_map, name='ForwardTTI', **kwargs)


# def ForwardOperatorStagg(model, source, receiver, space_order=4,
#                          save=False, **kwargs):
#     """
#     Constructor method for the forward modelling operator in an acoustic media
#     :param model: :class:`Model` object containing the physical parameters
#     :param source: :class:`PointData` object containing the source geometry
#     :param receiver: :class:`PointData` object containing the acquisition geometry
#     :param space_order: Space discretization order
#     :param save: Saving flag, True saves all time steps, False only the three
#     """
#     dt = model.grid.time_dim.spacing
#
#     m, damp, epsilon, delta, theta, phi = (model.m, model.damp, model.epsilon,
#                                            model.delta, model.theta, model.phi)
#     rho = 1
#     ndim = model.grid.dim
#     s = model.grid.time_dim.spacing
#     damp_loc = (1 - damp)
#     if ndim == 3:
#         stagg_x = (0, 1, 0, 0)
#         stagg_z = (0, 0, 0, 1)
#         stagg_y = (0, 0, 1, 0)
#         x, y, z = model.grid.dimensions
#     else:
#         stagg_x = (0, 1, 0)
#         stagg_z = (0, 0, 1)
#         x, z = model.grid.dimensions
#     # Create symbols for forward wavefield, source and receivers
#     vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
#                       time_order=1, space_order=space_order)
#     vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
#                       time_order=1, space_order=space_order)
#
#     if model.grid.dim == 3:
#         vy = TimeFunction(name='vy', grid=model.grid, staggered=stagg_y,
#                           time_order=1, space_order=space_order)
#
#     pv = TimeFunction(name='u', grid=model.grid,
#                       save=source.nt if save else None,
#                       time_order=1, space_order=space_order)
#
#     ph = TimeFunction(name='v', grid=model.grid,
#                       save=source.nt if save else None,
#                       time_order=1, space_order=space_order)
#     # Stencils
#     phdx = staggered_diff(ph, dim=x, order=space_order, stagger=left,
#                           theta=theta, phi=phi)
#     u_vx = Eq(vx.forward, damp_loc * vx - damp_loc * s / rho * phdx)
#
#     pvdz = staggered_diff(pv, dim=z, order=space_order, stagger=left,
#                           theta=theta, phi=phi)
#     u_vz = Eq(vz.forward, damp_loc * vz - damp_loc * s / rho * pvdz)
#
#     dvx = staggered_diff(vx.forward, dim=x, order=space_order, stagger=right,
#                          theta=theta, phi=phi)
#     dvz = staggered_diff(vz.forward, dim=z, order=space_order, stagger=right,
#                          theta=theta, phi=phi)
#
#     u_vy = []
#     dvy = 0
#     if ndim == 3:
#         phdy = staggered_diff(ph, dim=y, order=space_order, stagger=left,
#                               theta=theta, phi=phi)
#         u_vy = [Eq(vy.forward, damp_loc * vy - damp_loc * s / rho * phdy)]
#         dvy = staggered_diff(vy.forward, dim=y, order=space_order, stagger=right,
#                              theta=theta, phi=phi)
#
#     pv_eq = Eq(pv.forward,
#                damp_loc * (pv - s * rho / m * (delta * (dvx + dvy) + dvz)))
#
#     ph_eq = Eq(ph.forward,
#                damp_loc * (ph - s * rho / m * (epsilon * (dvx + dvy) + delta * dvz)))
#
#     # Source symbol with input wavelet
#     src = PointSource(name='src', grid=model.grid, time_range=source.time_range,
#                       npoint=source.npoint)
#     rec = Receiver(name='rec', grid=model.grid, time_range=receiver.time_range,
#                    npoint=receiver.npoint)
#     src_term = src.inject(field=pv.forward, offset=model.nbpml,
#                           expr=src * rho * dt / m)
#     src_term += src.inject(field=ph.forward, offset=model.nbpml,
#                            expr=src * rho * dt / m)
#     # Data is sampled at receiver locations
#     rec_term = rec.interpolate(expr=pv + ph, offset=model.nbpml)
#
#     # Substitute spacing terms to reduce flops
#     op = Operator([u_vx, u_vz] + u_vy + rec_term + [pv_eq, ph_eq] + src_term,
#                   subs=model.spacing_map, **kwargs)
#
#     return op


kernels = {('shifted', 3): kernel_shifted_3d, ('shifted', 2): kernel_shifted_2d,
           ('centered', 3): kernel_centered_3d, ('centered', 2): kernel_centered_2d,
           ('staggered', 3): kernel_staggered_3d, ('staggered', 2): kernel_staggered_2d}

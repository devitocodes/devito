from sympy import cos, sin, sqrt

from devito import Eq, Operator, TimeFunction, NODE
from examples.seismic import PointSource, Receiver
from devito.finite_differences import centered, first_derivative, transpose


def second_order_stencil(model, u, v, H0, Hz, forward=True):
    """
    Creates the stencil corresponding to the second order TTI wave equation
    u.dt2 =  (epsilon * H0 + delta * Hz) - damp * u.dt
    v.dt2 =  (delta * H0 + Hz) - damp * v.dt
    """

    m, damp = model.m, model.damp
    s = model.grid.stepping_dim.spacing

    unext = u.forward if forward else u.backward
    vnext = v.forward if forward else v.backward
    uprev = u.backward if forward else u.forward
    vprev = v.backward if forward else v.forward

    # Stencils
    stencilp = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * u + (s * damp - 2.0 * m) *
         uprev + 2.0 * s ** 2 * (H0))
    stencilr = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * v + (s * damp - 2.0 * m) *
         vprev + 2.0 * s ** 2 * (Hz))
    first_stencil = Eq(unext, stencilp)
    second_stencil = Eq(vnext, stencilr)

    stencils = [first_stencil, second_stencil]
    return stencils


def Gzz_centered(model, field, costheta, sintheta, cosphi, sinphi, space_order):
    """
    3D rotated second order derivative in the direction z.

    Parameters
    ----------
    field : Function
        Input for which the derivative is computed.
    costheta : Function or float
        Cosine of the tilt angle.
    sintheta : Function or float
        Sine of the tilt angle.
    cosphi : Function or float
        Cosine of the azymuth angle.
    sinphi : Function or float
        Sine of the azymuth angle.
    space_order : int
        Space discretization order.

    Returns
    -------
    Rotated second order derivative w.r.t. z.
    """
    order1 = space_order // 2
    x, y, z = model.space_dimensions
    Gz = -(sintheta * cosphi * first_derivative(field, dim=x,
                                                side=centered, fd_order=order1) +
           sintheta * sinphi * first_derivative(field, dim=y,
                                                side=centered, fd_order=order1) +
           costheta * first_derivative(field, dim=z,
                                       side=centered, fd_order=order1))

    Gzz = (first_derivative(Gz * sintheta * cosphi,
                            dim=x, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * sintheta * sinphi,
                            dim=y, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * costheta,
                            dim=z, side=centered, fd_order=order1,
                            matvec=transpose))
    return Gzz


def Gzz_centered_2d(model, field, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction z.

    Parameters
    ----------
    field : Function
        Input for which the derivative is computed.
    costheta : Function or float
        Cosine of the tilt angle.
    sintheta : Function or float
        Sine of the tilt angle.
    space_order : int
        Space discretization order.

    Returns
    -------
    Rotated second order derivative w.r.t. z.
    """
    order1 = space_order // 2
    x, y = model.space_dimensions[:2]
    Gz = -(sintheta * first_derivative(field, dim=x, side=centered, fd_order=order1) +
           costheta * first_derivative(field, dim=y, side=centered, fd_order=order1))
    Gzz = (first_derivative(Gz * sintheta, dim=x,
                            side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * costheta, dim=y,
                            side=centered, fd_order=order1,
                            matvec=transpose))
    return Gzz


# Centered case produces directly Gxx + Gyy
def Gxxyy_centered(model, field, costheta, sintheta, cosphi, sinphi, space_order):
    """
    Sum of the 3D rotated second order derivative in the direction x and y.
    As the Laplacian is rotation invariant, it is computed as the conventional
    Laplacian minus the second order rotated second order derivative in the direction z
    Gxx + Gyy = field.laplace - Gzz

    Parameters
    ----------
    field : Function
        Input field.
    costheta : Function or float
        Cosine of the tilt angle.
    sintheta : Function or float
        Sine of the tilt angle.
    cosphi : Function or float
        Cosine of the azymuth angle.
    sinphi : Function or float
        Sine of the azymuth angle.
    space_order : int
        Space discretization order.

    Returns
    -------
    Sum of the 3D rotated second order derivative in the direction x and y.
    """
    Gzz = Gzz_centered(model, field, costheta, sintheta, cosphi, sinphi, space_order)
    return field.laplace - Gzz


def Gxx_centered_2d(model, field, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction x.
    As the Laplacian is rotation invariant, it is computed as the conventional
    Laplacian minus the second order rotated second order derivative in the direction z
    Gxx = field.laplace - Gzz

    Parameters
    ----------
    field : TimeFunction
        Input field.
    costheta : Function or float
        Cosine of the tilt angle.
    sintheta : Function or float
        Sine of the tilt angle.
    cosphi : Function or float
        Cosine of the azymuth angle.
    sinphi : Function or float
        Sine of the azymuth angle.
    space_order : int
        Space discretization order.

    Returns
    -------
    Sum of the 3D rotated second order derivative in the direction x.
    """
    return field.laplace - Gzz_centered_2d(model, field, costheta, sintheta, space_order)


def kernel_centered_2d(model, u, v, space_order, forward=True):
    """
    TTI finite difference kernel. The equation solved is:
    
    u.dt2 = H0
    v.dt2 = Hz
    
    where H0 and Hz are defined as:

    H0 = (1+2 *epsilon) (Gxx(u)+Gyy(u)) + sqrt(1+ 2*delta) Gzz(v)
    Hz = sqrt(1+ 2*delta) (Gxx(u)+Gyy(u)) +  Gzz(v)
    
    and
    
    H0 = (Gxx+Gyy)((1+2 *epsilon)*u + sqrt(1+ 2*delta)*v)
    Hz = Gzz(sqrt(1+ 2*delta)*u + v)
    
    for the forward and adjoint cases, respectively. Epsilon and delta are the Thomsen
    parameters. This function computes H0 and Hz.
    
    References
    ----------
    Zhang, Yu, Houzhu Zhang, and Guanquan Zhang. "A stable TTI reverse time migration and
    its implementation." Geophysics 76.3 (2011): WA3-WA11.
    
    Louboutin, Mathias, Philipp Witte, and Felix J. Herrmann. "Effects of wrong adjoints
    for RTM in TTI media." SEG Technical Program Expanded Abstracts 2018. Society of
    Exploration Geophysicists, 2018. 331-335.

    Parameters
    ----------
    u : TimeFunction
        First TTI field.
    v : TimeFunction
        Second TTI field.
    space_order : int
        Space discretization order.

    Returns
    -------
    u and v component of the rotated Laplacian in 2D.
    """
    # Tilt and azymuth setup
    costheta = cos(model.theta)
    sintheta = sin(model.theta)

    delta, epsilon = model.delta, model.epsilon
    epsilon = 1 + 2*epsilon
    delta = sqrt(1 + 2*delta)

    if forward:
        Gxx = Gxx_centered_2d(model, u, costheta, sintheta, space_order)
        Gzz = Gzz_centered_2d(model, v, costheta, sintheta, space_order)
        H0 = epsilon*Gxx + delta*Gzz
        Hz = delta*Gxx + Gzz
        return second_order_stencil(model, u, v, H0, Hz)
    else:
        H0 = Gxx_centered_2d(model, (epsilon*u + delta*v), costheta,
                             sintheta, space_order)
        Hz = Gzz_centered_2d(model, (delta*u + v), costheta, sintheta, space_order)
        return second_order_stencil(model, u, v, H0, Hz, forward=False)


def kernel_centered_3d(model, u, v, space_order, forward=True):
    """
    TTI finite difference kernel. The equation solved is:
    
    u.dt2 = H0
    v.dt2 = Hz
    
    where H0 and Hz are defined as:

    H0 = (1+2 *epsilon) (Gxx(u)+Gyy(u)) + sqrt(1+ 2*delta) Gzz(v)
    Hz = sqrt(1+ 2*delta) (Gxx(u)+Gyy(u)) +  Gzz(v)
    
    and
    
    H0 = (Gxx+Gyy)((1+2 *epsilon)*u + sqrt(1+ 2*delta)*v)
    Hz = Gzz(sqrt(1+ 2*delta)*u + v)
    
    for the forward and adjoint cases, respectively. Epsilon and delta are the Thomsen
    parameters. This function computes H0 and Hz.
    
    References
    ----------
    Zhang, Yu, Houzhu Zhang, and Guanquan Zhang. "A stable TTI reverse time migration and
    its implementation." Geophysics 76.3 (2011): WA3-WA11.
    
    Louboutin, Mathias, Philipp Witte, and Felix J. Herrmann. "Effects of wrong adjoints
    for RTM in TTI media." SEG Technical Program Expanded Abstracts 2018. Society of
    Exploration Geophysicists, 2018. 331-335.

    Parameters
    ----------
    u : TimeFunction
        First TTI field.
    v : TimeFunction
        Second TTI field.
    space_order : int
        Space discretization order.

    Returns
    -------
    u and v component of the rotated Laplacian in 3D.
    """
    # Tilt and azymuth setup
    costheta = cos(model.theta)
    sintheta = sin(model.theta)
    cosphi = cos(model.phi)
    sinphi = sin(model.phi)

    delta, epsilon = model.delta, model.epsilon
    epsilon = 1 + 2*epsilon
    delta = sqrt(1 + 2*delta)

    if forward:
        Gxx = Gxxyy_centered(model, u, costheta, sintheta, cosphi, sinphi, space_order)
        Gzz = Gzz_centered(model, v, costheta, sintheta, cosphi, sinphi, space_order)
        H0 = epsilon*Gxx + delta*Gzz
        Hz = delta*Gxx + Gzz
        return second_order_stencil(model, u, v, H0, Hz)
    else:
        H0 = Gxxyy_centered(model, (epsilon*u + delta*v), costheta, sintheta,
                            cosphi, sinphi, space_order)
        Hz = Gzz_centered(model, (delta*u + v), costheta, sintheta, cosphi,
                          sinphi, space_order)
        return second_order_stencil(model, u, v, H0, Hz, forward=False)


def particle_velocity_fields(model, space_order):
    """
    Initialize particle velocity fields for staggered TTI.
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_x = x
        stagg_z = z
        x, z = model.grid.dimensions
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order)
        vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order)
        vy = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_x = x
        stagg_y = y
        stagg_z = z
        x, y, z = model.grid.dimensions
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order)
        vy = TimeFunction(name='vy', grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order)
        vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order)

    return vx, vz, vy


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
    epsilon = 1 + 2 * epsilon
    delta = sqrt(1 + 2 * delta)
    s = model.grid.stepping_dim.spacing
    x, z = model.grid.dimensions
    # Staggered setup
    vx, vz, _ = particle_velocity_fields(model, space_order)

    # Stencils
    phdx = cos(theta) * u.dx - sin(theta) * u.dy
    u_vx = Eq(vx.forward, dampl * vx - dampl * s * phdx)

    pvdz = sin(theta) * v.dx + cos(theta) * v.dy
    u_vz = Eq(vz.forward, dampl * vz - dampl * s * pvdz)

    dvx = cos(theta) * vx.forward.dx - sin(theta) * vx.forward.dy
    dvz = sin(theta) * vz.forward.dx + cos(theta) * vz.forward.dy

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
    epsilon = 1 + 2 * epsilon
    delta = sqrt(1 + 2 * delta)
    s = model.grid.stepping_dim.spacing
    x, y, z = model.grid.dimensions
    # Staggered setup
    vx, vz, vy = particle_velocity_fields(model, space_order)
    # Stencils
    phdx = (cos(theta) * cos(phi) * u.dx +
            cos(theta) * sin(phi) * u.dyc -
            sin(theta) * u.dzc)
    u_vx = Eq(vx.forward, dampl * vx - dampl * s * phdx)

    phdy = -sin(phi) * u.dxc + cos(phi) * u.dy
    u_vy = Eq(vy.forward, dampl * vy - dampl * s * phdy)

    pvdz = (sin(theta) * cos(phi) * v.dxc +
            sin(theta) * sin(phi) * v.dyc +
            cos(theta) * v.dz)
    u_vz = Eq(vz.forward, dampl * vz - dampl * s * pvdz)

    dvx = (cos(theta) * cos(phi) * vx.forward.dx +
           cos(theta) * sin(phi) * vx.forward.dyc -
           sin(theta) * vx.forward.dzc)
    dvy = -sin(phi) * vy.forward.dxc + cos(phi) * vy.forward.dy
    dvz = (sin(theta) * cos(phi) * vz.forward.dxc +
           sin(theta) * sin(phi) * vz.forward.dyc +
           cos(theta) * vz.forward.dz)
    # u and v equations
    pv_eq = Eq(v.forward, dampl * (v - s / m * (delta * (dvx + dvy) + dvz)))

    ph_eq = Eq(u.forward, dampl * (u - s / m * (epsilon * (dvx + dvy) +
                                                delta * dvz)))

    return [u_vx, u_vy, u_vz] + [pv_eq, ph_eq]


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='centered', **kwargs):
    """
    Construct an forward modelling operator in an tti media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    kernel : str, optional
        Type of discretization, centered or shifted
    """

    dt = model.grid.time_dim.spacing
    m = model.m
    time_order = 1 if kernel == 'staggered' else 2
    if kernel == 'staggered':
        stagg_u = stagg_v = NODE
    else:
        stagg_u = stagg_v = None

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid, staggered=stagg_u,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, staggered=stagg_v,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # FD kernels of the PDE
    FD_kernel = kernels[(kernel, len(model.shape))]
    stencils = FD_kernel(model, u, v, space_order)

    # Source and receivers
    stencils += src.inject(field=u.forward, expr=src * dt**2 / m)
    stencils += src.inject(field=v.forward, expr=src * dt**2 / m)
    stencils += rec.interpolate(expr=u + v)

    # Substitute spacing terms to reduce flops
    return Operator(stencils, subs=model.spacing_map, name='ForwardTTI', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    **kwargs):
    """
    Construct an adjoint modelling operator in an tti media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    """

    dt = model.grid.time_dim.spacing
    m = model.m
    time_order = 2

    # Create symbols for forward wavefield, source and receivers
    p = TimeFunction(name='p', grid=model.grid, save=None, time_order=time_order,
                     space_order=space_order)
    r = TimeFunction(name='r', grid=model.grid, save=None, time_order=time_order,
                     space_order=space_order)
    srca = PointSource(name='srca', grid=model.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # FD kernels of the PDE
    FD_kernel = kernels[('centered', len(model.shape))]
    stencils = FD_kernel(model, p, r, space_order, forward=False)

    # Construct expression to inject receiver values
    stencils += rec.inject(field=p.backward, expr=rec * dt**2 / m)
    stencils += rec.inject(field=r.backward, expr=rec * dt**2 / m)

    # Create interpolation expression for the adjoint-source
    stencils += srca.interpolate(expr=p + r)

    # Substitute spacing terms to reduce flops
    return Operator(stencils, subs=model.spacing_map, name='AdjointTTI', **kwargs)


kernels = {('centered', 3): kernel_centered_3d, ('centered', 2): kernel_centered_2d,
           ('staggered', 3): kernel_staggered_3d, ('staggered', 2): kernel_staggered_2d}

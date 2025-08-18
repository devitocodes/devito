from devito import (Eq, Operator, Function, TimeFunction, NODE, Inc, solve,
                    cos, sin, sqrt)
from examples.seismic.acoustic.operators import freesurface


def _subs(func, d, d0):
    try:
        return func._subs(d, d0)
    except AttributeError:
        return func


def second_order_stencil(model, u, v, H0, Hz, qu, qv, forward=True):
    """
    Creates the stencil corresponding to the second order TTI wave equation
    m * u.dt2 =  (epsilon * H0 + delta * Hz) - damp * u.dt
    m * v.dt2 =  (delta * H0 + Hz) - damp * v.dt
    """
    m, damp = model.m, model.damp

    unext = u.forward if forward else u.backward
    vnext = v.forward if forward else v.backward
    udt = u.dt if forward else u.dt.T
    vdt = v.dt if forward else v.dt.T

    # Stencils
    stencilp = solve(m * u.dt2 - H0 - qu + damp * udt, unext)
    stencilr = solve(m * v.dt2 - Hz - qv + damp * vdt, vnext)

    first_stencil = Eq(unext, stencilp, subdomain=model.grid.subdomains['physdomain'])
    second_stencil = Eq(vnext, stencilr, subdomain=model.grid.subdomains['physdomain'])

    stencils = [first_stencil, second_stencil]

    # Add free surface
    if model.fs:
        stencils.append(freesurface(model, Eq(unext, stencilp)))
        stencils.append(freesurface(model, Eq(vnext, stencilr)))

    return stencils


def trig_func(model):
    """
    Trigonometric function of the tilt and azymuth angles.
    """
    try:
        theta = model.theta
    except AttributeError:
        theta = 0

    costheta = cos(theta)
    sintheta = sin(theta)
    if model.dim == 3:
        try:
            phi = model.phi
        except AttributeError:
            phi = 0

        cosphi = cos(phi)
        sinphi = sin(phi)
        return costheta, sintheta, cosphi, sinphi
    return costheta, sintheta


def Gzz_centered(model, field):
    """
    3D rotated second order derivative in the direction z.

    Parameters
    ----------
    model : Model
        Physical parameters model structure.
    field : Function
        Input for which the derivative is computed.

    Returns
    -------
    Rotated second order derivative w.r.t. z.
    """
    b = getattr(model, 'b', 1)
    costheta, sintheta, cosphi, sinphi = trig_func(model)

    order1 = field.space_order // 2
    x, y, z = field.grid.dimensions
    dx, dy, dz = x.spacing/2, y.spacing/2, z.spacing/2

    Gz = (_subs(b, x, x+dx) * sintheta * cosphi * field.dx(fd_order=order1, x0=x+dx) +
          _subs(b, y, y+dy) * sintheta * sinphi * field.dy(fd_order=order1, x0=y+dy) +
          _subs(b, z, z+dz) * costheta * field.dz(fd_order=order1, x0=z+dz))

    Gzz = (Gz * costheta).dz(fd_order=order1, x0=z-dz)
    # Add rotated derivative if angles are not zero. If angles are
    # zeros then `0*Gz = 0` and doesn't have any `.dy` ....
    if sintheta != 0:
        Gzz += (Gz * sintheta * cosphi).dx(fd_order=order1, x0=x-dx)
    if sinphi != 0:
        Gzz += (Gz * sintheta * sinphi).dy(fd_order=order1, x0=y-dy)

    return Gzz


def Gzz_centered_2d(model, field):
    """
    2D rotated second order derivative in the direction z.

    Parameters
    ----------
    model : Model
        Physical parameters model structure.
    field : Function
        Input for which the derivative is computed.

    Returns
    -------
    Rotated second order derivative w.r.t. z.
    """
    b = getattr(model, 'b', 1)
    costheta, sintheta = trig_func(model)

    order1 = field.space_order // 2
    x, y = field.grid.dimensions
    dx, dy = x.spacing/2, y.spacing/2

    Gz = (_subs(b, x, x+dx) * sintheta * field.dx(fd_order=order1, x0=x+dx) +
          _subs(b, y, y+dy) * costheta * field.dy(fd_order=order1, x0=y+dy))
    Gzz = (Gz * costheta).dy(fd_order=order1, x0=y-dy)

    # Add rotated derivative if angles are not zero. If angles are
    # zeros then `0*Gz = 0` and doesn't have any `.dy` ....
    if sintheta != 0:
        Gzz += (Gz * sintheta).dx(fd_order=order1, x0=x-dx)
    return Gzz


# Centered case produces directly Gxx + Gyy
def Gh_centered(model, field):
    """
    Sum of the 3D rotated second order derivative in the direction x and y.
    As the Laplacian is rotation invariant, it is computed as the conventional
    Laplacian minus the second order rotated second order derivative in the direction z
    Gxx + Gyy = field.laplace - Gzz

    Parameters
    ----------
    model : Model
        Physical parameters model structure.
    field : Function
        Input field.

    Returns
    -------
    Sum of the 3D rotated second order derivative in the direction x and y.
    """
    if model.dim == 3:
        Gzz = Gzz_centered(model, field)
    else:
        Gzz = Gzz_centered_2d(model, field)
    b = getattr(model, 'b', None)
    if b is not None:
        _diff = lambda f, d: getattr(f, f'd{d.name}')
        so = field.space_order // 2
        lap = 0
        for d in model.space_dimensions:
            x0 = d + d.spacing / 2
            x0m = d - d.spacing / 2
            lap += _diff(_subs(b, d, x0) * _diff(field, d)(x0=x0, fd_order=so),
                         d)(x0=x0m, fd_order=so)
    else:
        lap = field.laplace
    return lap - Gzz


def kernel_centered(model, u, v, **kwargs):
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

    References:
        * Zhang, Yu, Houzhu Zhang, and Guanquan Zhang. "A stable TTI reverse
          time migration and its implementation." Geophysics 76.3 (2011): WA3-WA11.
        * Louboutin, Mathias, Philipp Witte, and Felix J. Herrmann. "Effects of
          wrong adjoints for RTM in TTI media." SEG Technical Program Expanded
          Abstracts 2018. Society of Exploration Geophysicists, 2018. 331-335.

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
    # Forward or backward
    forward = kwargs.get('forward', True)

    delta, epsilon = model.delta, model.epsilon
    epsilon = 1 + 2*epsilon
    delta = sqrt(1 + 2*delta)

    # Get source
    qu = kwargs.get('qu', 0)
    qv = kwargs.get('qv', 0)

    Gzz = Gzz_centered_2d if model.dim == 2 else Gzz_centered

    if forward:
        Gxx = Gh_centered(model, u)
        Gzz = Gzz(model, v)
        H0 = epsilon*Gxx + delta*Gzz
        Hz = delta*Gxx + Gzz
        return second_order_stencil(model, u, v, H0, Hz, qu, qv)
    else:
        H0 = Gh_centered(model, (epsilon*u + delta*v))
        Hz = Gzz(model, (delta*u + v))
        return second_order_stencil(model, u, v, H0, Hz, qu, qv, forward=forward)


def particle_velocity_fields(model, space_order):
    """
    Initialize particle velocity fields for staggered TTI.
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_x = x
        stagg_z = z
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
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order)
        vy = TimeFunction(name='vy', grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order)
        vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order)

    return vx, vz, vy


def kernel_staggered_2d(model, u, v, **kwargs):
    """
    TTI finite difference. The equation solved is:

    vx.dt = - u.dx
    vz.dt = - v.dx
    m * v.dt = - sqrt(1 + 2 delta) vx.dx - vz.dz + Fh
    m * u.dt = - (1 + 2 epsilon) vx.dx - sqrt(1 + 2 delta) vz.dz + Fv
    """
    space_order = u.space_order
    # Forward or backward
    forward = kwargs.get('forward', True)

    dampl = 1 - model.damp
    m, epsilon, delta = model.m, model.epsilon, model.delta
    costheta, sintheta = trig_func(model)
    epsilon = 1 + 2 * epsilon
    delta = sqrt(1 + 2 * delta)
    s = model.grid.stepping_dim.spacing
    x, z = model.grid.dimensions

    # Get source
    qu = kwargs.get('qu', 0)
    qv = kwargs.get('qv', 0)

    # Staggered setup
    vx, vz, _ = particle_velocity_fields(model, space_order)

    if forward:
        # Stencils
        phdx = costheta * u.dx - sintheta * u.dyc
        u_vx = Eq(vx.forward, dampl * vx - dampl * s * phdx)

        pvdz = sintheta * v.dxc + costheta * v.dy
        u_vz = Eq(vz.forward, dampl * vz - dampl * s * pvdz)

        dvx = costheta * vx.forward.dx - sintheta * vx.forward.dyc
        dvz = sintheta * vz.forward.dxc + costheta * vz.forward.dy

        # u and v equations
        pv_eq = Eq(v.forward, dampl * (v - s / m * (delta * dvx + dvz)) + s / m * qv)
        ph_eq = Eq(u.forward, dampl * (u - s / m * (epsilon * dvx + delta * dvz)) +
                   s / m * qu)
    else:
        # Stencils
        phdx = ((costheta*epsilon*u).dx - (sintheta*epsilon*u).dyc +
                (costheta*delta*v).dx - (sintheta*delta*v).dyc)
        u_vx = Eq(vx.backward, dampl * vx + dampl * s * phdx)

        pvdz = ((sintheta*delta*u).dxc + (costheta*delta*u).dy +
                (sintheta*v).dxc + (costheta*v).dy)
        u_vz = Eq(vz.backward, dampl * vz + dampl * s * pvdz)

        dvx = (costheta * vx.backward).dx - (sintheta * vx.backward).dyc
        dvz = (sintheta * vz.backward).dxc + (costheta * vz.backward).dy

        # u and v equations
        pv_eq = Eq(v.backward, dampl * (v + s / m * dvz))
        ph_eq = Eq(u.backward, dampl * (u + s / m * dvx))

    return [u_vx, u_vz] + [pv_eq, ph_eq]


def kernel_staggered_3d(model, u, v, **kwargs):
    """
    TTI finite difference. The equation solved is:

    vx.dt = - u.dx
    vy.dt = - u.dx
    vz.dt = - v.dx
    m * v.dt = - sqrt(1 + 2 delta) (vx.dx + vy.dy) - vz.dz + Fh
    m * u.dt = - (1 + 2 epsilon) (vx.dx + vy.dy) - sqrt(1 + 2 delta) vz.dz + Fv
    """
    space_order = u.space_order
    # Forward or backward
    forward = kwargs.get('forward', True)

    dampl = 1 - model.damp
    m, epsilon, delta = model.m, model.epsilon, model.delta
    costheta, sintheta, cosphi, sinphi = trig_func(model)
    epsilon = 1 + 2 * epsilon
    delta = sqrt(1 + 2 * delta)
    s = model.grid.stepping_dim.spacing
    x, y, z = model.grid.dimensions

    # Get source
    qu = kwargs.get('qu', 0)
    qv = kwargs.get('qv', 0)

    # Staggered setup
    vx, vz, vy = particle_velocity_fields(model, space_order)

    if forward:
        # Stencils
        phdx = (costheta * cosphi * u.dx +
                costheta * sinphi * u.dyc -
                sintheta * u.dzc)
        u_vx = Eq(vx.forward, dampl * vx - dampl * s * phdx)

        phdy = -sinphi * u.dxc + cosphi * u.dy
        u_vy = Eq(vy.forward, dampl * vy - dampl * s * phdy)

        pvdz = (sintheta * cosphi * v.dxc +
                sintheta * sinphi * v.dyc +
                costheta * v.dz)
        u_vz = Eq(vz.forward, dampl * vz - dampl * s * pvdz)

        dvx = (costheta * cosphi * vx.forward.dx +
               costheta * sinphi * vx.forward.dyc -
               sintheta * vx.forward.dzc)
        dvy = -sinphi * vy.forward.dxc + cosphi * vy.forward.dy
        dvz = (sintheta * cosphi * vz.forward.dxc +
               sintheta * sinphi * vz.forward.dyc +
               costheta * vz.forward.dz)
        # u and v equations
        pv_eq = Eq(v.forward, dampl * (v - s / m * (delta * (dvx + dvy) + dvz)) +
                   s / m * qv)

        ph_eq = Eq(u.forward, dampl * (u - s / m * (epsilon * (dvx + dvy) +
                                                    delta * dvz)) + s / m * qu)
    else:
        # Stencils
        phdx = ((costheta * cosphi * epsilon*u).dx +
                (costheta * sinphi * epsilon*u).dyc -
                (sintheta * epsilon*u).dzc + (costheta * cosphi * delta*v).dx +
                                             (costheta * sinphi * delta*v).dyc -
                                             (sintheta * delta*v).dzc)
        u_vx = Eq(vx.backward, dampl * vx + dampl * s * phdx)

        phdy = (-(sinphi * epsilon*u).dxc + (cosphi * epsilon*u).dy -
                (sinphi * delta*v).dxc + (cosphi * delta*v).dy)
        u_vy = Eq(vy.backward, dampl * vy + dampl * s * phdy)

        pvdz = ((sintheta * cosphi * delta*u).dxc +
                (sintheta * sinphi * delta*u).dyc +
                (costheta * delta*u).dz + (sintheta * cosphi * v).dxc +
                                          (sintheta * sinphi * v).dyc +
                                          (costheta * v).dz)
        u_vz = Eq(vz.backward, dampl * vz + dampl * s * pvdz)

        dvx = ((costheta * cosphi * vx.backward).dx +
               (costheta * sinphi * vx.backward).dyc -
               (sintheta * vx.backward).dzc)
        dvy = (-sinphi * vy.backward).dxc + (cosphi * vy.backward).dy
        dvz = ((sintheta * cosphi * vz.backward).dxc +
               (sintheta * sinphi * vz.backward).dyc +
               (costheta * vz.backward).dz)
        # u and v equations
        pv_eq = Eq(v.backward, dampl * (v + s / m * dvz))

        ph_eq = Eq(u.backward, dampl * (u + s / m * (dvx + dvy)))

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
    src = geometry.src
    rec = geometry.rec

    # FD kernels of the PDE
    FD_kernel = kernels[(kernel, len(model.shape))]
    stencils = FD_kernel(model, u, v)

    # Source and receivers
    expr = src * dt / m if kernel == 'staggered' else src * dt**2 / m
    stencils += src.inject(field=(u.forward, v.forward), expr=expr)
    stencils += rec.interpolate(expr=u + v)

    # Substitute spacing terms to reduce flops
    return Operator(stencils, subs=model.spacing_map, name='ForwardTTI', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    kernel='centered', **kwargs):
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
    kernel : str, optional
        Type of discretization, centered or shifted
    """

    dt = model.grid.time_dim.spacing
    m = model.m
    time_order = 1 if kernel == 'staggered' else 2
    if kernel == 'staggered':
        stagg_p = stagg_r = NODE
    else:
        stagg_p = stagg_r = None

    # Create symbols for forward wavefield, source and receivers
    p = TimeFunction(name='p', grid=model.grid, staggered=stagg_p,
                     time_order=time_order, space_order=space_order)
    r = TimeFunction(name='r', grid=model.grid, staggered=stagg_r,
                     time_order=time_order, space_order=space_order)
    srca = geometry.new_src(name='srca', src_type=None)
    rec = geometry.rec

    # FD kernels of the PDE
    FD_kernel = kernels[(kernel, len(model.shape))]
    stencils = FD_kernel(model, p, r, forward=False)

    # Construct expression to inject receiver values
    expr = rec * dt / m if kernel == 'staggered' else rec * dt**2 / m
    stencils += rec.inject(field=(p.backward, r.backward), expr=expr)

    # Create interpolation expression for the adjoint-source
    stencils += srca.interpolate(expr=p + r)

    # Substitute spacing terms to reduce flops
    return Operator(stencils, subs=model.spacing_map, name='AdjointTTI', **kwargs)


def JacobianOperator(model, geometry, space_order=4,
                     **kwargs):
    """
    Construct a Linearized Born operator in a TTI media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or staggered.
    """
    dt = model.grid.stepping_dim.spacing
    m = model.m
    time_order = 2

    # Create source and receiver symbols
    src = geometry.src
    rec = geometry.rec

    # Create wavefields and a dm field
    u0 = TimeFunction(name='u0', grid=model.grid, save=None, time_order=time_order,
                      space_order=space_order)
    v0 = TimeFunction(name='v0', grid=model.grid, save=None, time_order=time_order,
                      space_order=space_order)
    du = TimeFunction(name="du", grid=model.grid, save=None,
                      time_order=2, space_order=space_order)
    dv = TimeFunction(name="dv", grid=model.grid, save=None,
                      time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    # FD kernels of the PDE
    FD_kernel = kernels[('centered', len(model.shape))]
    eqn1 = FD_kernel(model, u0, v0)

    # Linearized source and stencil
    lin_usrc = -dm * u0.dt2
    lin_vsrc = -dm * v0.dt2

    eqn2 = FD_kernel(model, du, dv, qu=lin_usrc, qv=lin_vsrc)

    # Construct expression to inject source values, injecting at u0(t+dt)/v0(t+dt)
    src_term = src.inject(field=(u0.forward, v0.forward), expr=src * dt**2 / m)

    # Create interpolation expression for receivers, extracting at du(t)+dv(t)
    rec_term = rec.interpolate(expr=du + dv)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + src_term + eqn2 + rec_term, subs=model.spacing_map,
                    name='BornTTI', **kwargs)


def JacobianAdjOperator(model, geometry, space_order=4,
                        save=True, **kwargs):
    """
    Construct a linearized JacobianAdjoint modeling Operator in a TTI media.

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
        Option to store the entire (unrolled) wavefield.
    """
    dt = model.grid.stepping_dim.spacing
    m = model.m
    time_order = 2

    # Gradient symbol and wavefield symbols
    u0 = TimeFunction(name='u0', grid=model.grid, save=geometry.nt if save
                      else None, time_order=time_order, space_order=space_order)
    v0 = TimeFunction(name='v0', grid=model.grid, save=geometry.nt if save
                      else None, time_order=time_order, space_order=space_order)

    du = TimeFunction(name="du", grid=model.grid, save=None,
                      time_order=time_order, space_order=space_order)
    dv = TimeFunction(name="dv", grid=model.grid, save=None,
                      time_order=time_order, space_order=space_order)

    dm = Function(name="dm", grid=model.grid)

    rec = geometry.rec

    # FD kernels of the PDE
    FD_kernel = kernels[('centered', len(model.shape))]
    eqn = FD_kernel(model, du, dv, forward=False)

    dm_update = Inc(dm, - (u0 * du.dt2 + v0 * dv.dt2))

    # Add expression for receiver injection
    rec_term = rec.inject(field=(du.backward, dv.backward), expr=rec * dt**2 / m)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + rec_term + [dm_update], subs=model.spacing_map,
                    name='GradientTTI', **kwargs)


kernels = {('centered', 3): kernel_centered, ('centered', 2): kernel_centered,
           ('staggered', 3): kernel_staggered_3d, ('staggered', 2): kernel_staggered_2d}

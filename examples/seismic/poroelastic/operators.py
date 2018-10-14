from devito import Eq, Operator, TimeFunction, NODE
from examples.seismic import PointSource, Receiver


def stress_fields(model, save, space_order):
    """
    Create the TimeFunction objects for the stress fields in the poroelastic formulation
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_xx = stagg_zz = NODE
        stagg_xz = (x, z)
        # Create symbols for forward wavefield, source and receivers
        txx = TimeFunction(name='txx', grid=model.grid, staggered=stagg_xx, save=save,
                           time_order=1, space_order=space_order)
        tzz = TimeFunction(name='tzz', grid=model.grid, staggered=stagg_zz, save=save,
                           time_order=1, space_order=space_order)
        txz = TimeFunction(name='txz', grid=model.grid, staggered=stagg_xz, save=save,
                           time_order=1, space_order=space_order)
        tyy = txy = tyz = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_xx = stagg_yy = stagg_zz = NODE
        stagg_xz = (x, z)
        stagg_yz = (y, z)
        stagg_xy = (x, y)
        # Create symbols for forward wavefield, source and receivers
        txx = TimeFunction(name='txx', grid=model.grid, staggered=stagg_xx, save=save,
                           time_order=1, space_order=space_order)
        tzz = TimeFunction(name='tzz', grid=model.grid, staggered=stagg_zz, save=save,
                           time_order=1, space_order=space_order)
        tyy = TimeFunction(name='tyy', grid=model.grid, staggered=stagg_yy, save=save,
                           time_order=1, space_order=space_order)
        txz = TimeFunction(name='txz', grid=model.grid, staggered=stagg_xz, save=save,
                           time_order=1, space_order=space_order)
        txy = TimeFunction(name='txy', grid=model.grid, staggered=stagg_xy, save=save,
                           time_order=1, space_order=space_order)
        tyz = TimeFunction(name='tyz', grid=model.grid, staggered=stagg_yz, save=save,
                           time_order=1, space_order=space_order)

    return txx, tyy, tzz, txy, txz, tyz
    

def pressure_fields(model, save, space_order):
    """
    Create the TimeFunction objects for the pressure fields in the poroelastic formulation
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_p = NODE
        # Create symbols for forward wavefield, source and receivers
        p = TimeFunction(name='p', grid=model.grid, staggered=stagg_p, save=save,
                           time_order=1, space_order=space_order)
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_p = NODE
        # Create symbols for forward wavefield, source and receivers
        p = TimeFunction(name='p', grid=model.grid, staggered=stagg_p, save=save,
                           time_order=1, space_order=space_order)
    return p


def particle_velocity_fields(model, save, space_order):
    """
    Create the particle velocity fields
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_x = x
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order, save=save)
        vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order, save=save)
        vy = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_x = x
        stagg_y = y
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order, save=save)
        vy = TimeFunction(name='vy', grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order, save=save)
        vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order, save=save)

    return vx, vy, vz
# ------------------------------------------------------------------------------

def relative_velocity_fields(model, save, space_order):
    """
    Create the relative velocity fields
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_x = x
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        wx = TimeFunction(name='wx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order, save=save)
        wz = TimeFunction(name='wz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order, save=save)
        wy = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_x = x
        stagg_y = y
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        wx = TimeFunction(name='wx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order, save=save)
        wy = TimeFunction(name='wy', grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order, save=save)
        wz = TimeFunction(name='wz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order, save=save)

    return wx, wy, wz
# ------------------------------------------------------------------------------

def poroelastic_2d(model, space_order, save, source, receiver):
    """
    2D poroelastic wave equation FD kernel
    """
    rho_s, rho_f, phi, k, mu_f, K_dr, K_s, K_f, G, T, damp = model.G, model.rho_s, model.rho_f, model.phi, model.k, model.mu_f, model.K_dr, model.K_s, model.K_f, model.T, model.damp
    
    # Delta T (sic)                                               
    dt = model.grid.stepping_dim.spacing
    
    # Biot Coefficient
    alpha = 1.0 - K_dr/K_s
    
    # Biot Modulus
    M = 1.0 / (phi/K_f + (alpha - phi)/K_s)
    
    # Bulk Density
    rho_b = phi*rho_f + (1.0 - phi)*rho_s
    
    # Saturated / Gassmann Bulk Modulus
    K_G = K_dr + alpha**2 * M

    # Lame Parameter of Saturated Medium
    l_c = K_G - 2/3 * G
    
    # Create symbols for forward wavefield, source and receivers
    vx, vy, vz = particle_velocity_fields(model, save, space_order)
    wx, wy, wz = relative_velocity_fields(model, save, space_order)    
    txx, tyy, tzz, _, txz, _ = stress_fields(model, save, space_order)
    p = pressure_fields(model, save, space_order) # Different order needed?
    
    # Convenience terms for nightmarish poroelastodynamic FD stencils
    _m = T * (rho_f/phi)  # Effective fluid density / mass coupling coefficient
    _b = mu_f / k         # Fluid Mobility / resistiving damping
    
    A = _m / (_m * rho_b - rho_f**2)
    B = (rho_b * _b) / (rho_b - rho_f**2)
    C = rho_f / (_m * rho_b - rho_f**2)
    D = (-1 * rho_f) / (_m * rho_b - rho_f**2)
    E = (-1 * rho_b * _b) / (_m * rho_b - rho_f**2)
    F = (-1 * rho_b) / (_m * rho_b - rho_f**2)
    
    # Stencils
    u_vx  = Eq(vx.forward, damp*(vx + dt*( A*(txx.dx + txz.dy) + C*p.dx + B*wx ) ) )
    u_vz  = Eq(vz.forward, damp*(vz + dt*( A*(txz.dx + tzz.dy) + C*p.dy + B*wz ) ) )
    
    u_wx  = Eq(wx.forward, damp*(wx + dt*( D*(txx.dx + txz.dy) + F*p.dx + E*wx ) ) )
    u_wz  = Eq(wz.forward, damp*(wz + dt*( D*(txz.dx + tzz.dy) + F*p.dy + E*wz ) ) )
    
    u_txx = Eq(txx.forward, damp*(txx + dt*((l_c + 2*G)*vx.forward.dx + l_c*vz.forward.dy
                                      + alpha*M*(wx.forward.dx + wz.forward.dy) ) ) )
    u_tzz = Eq(tzz.forward, damp*(tzz + dt*((l_c + 2*G)*vz.forward.dy + l_c*vx.forward.dx
                                      + alpha*M*(wx.forward.dx + wz.forward.dy) ) ) )
    u_txz = Eq(txz.forward, damp*(txz + dt*(G*(vz.forward.dx + vx.forward.dy) ) ) )
    
    u_p   = Eq(p.forward, damp*(p - dt*(alpha*M*(vx.forward.dx + vz.forward.dy) 
                              + M*(wx.forward.dx + wz.forward.dy) ) ) )

    src_rec_expr = src_rec(vx, vy, vz, txx, tyy, tzz, model, source, receiver)
    return [u_vx, u_vz, u_wx, u_wz, u_txx, u_tzz, u_txz, u_p] + src_rec_expr
# ------------------------------------------------------------------------------

def poroelastic_3d(model, space_order, save, source, receiver):
    """
    3D elastic wave equation FD kernel
    """
    vp, vs, rho_s, rho_f, phi, k, mu_f, K_dr, K_s, K_f, damp = model.vp, model.vs, model.rho_s, model.rho_f, model.phi, model.k, model.mu_f, model.K_dr, model.K_s, model.K_f, model.damp
    
    dt = model.grid.stepping_dim.spacing
    
    # Biot Coefficient
    alpha = 1.0 - K_dr/K_s
    
    # Biot Modulus
    M = phi/K_f + (alpha - phi)/K_s
    
    # Bulk Density
    rho_b = phi*rho_f + (1.0 - phi)*rho_s

    # Shear Modulus of Sa                mturated Rock
    mu = (vs**2)*rho_b
    
    # Lame Parameter of Saturated Rock
    l = rho_b*(vp**2 - 2*(vs**2))

    # Create symbols for forward wavefield, source and receivers
    vx, vy, vz = particle_velocity_fields(model, save, space_order)
    wx, wy, wz = relative_velocity_fields(model, save, space_order)    
    txx, tyy, tzz, txy, txz, tyz = stress_fields(model, save, space_order)

    # Stencils
    u_vx = Eq(vx.forward, damp * vx - damp * dt * 1.0/rho_b * (txx.dx + txy.dy + txz.dz))
    u_vy = Eq(vy.forward, damp * vy - damp * dt * 1.0/rho_b * (txy.dx + tyy.dy + tyz.dz))
    u_vz = Eq(vz.forward, damp * vz - damp * dt * 1.0/rho_b * (txz.dx + tyz.dy + tzz.dz))

    u_txx = Eq(txx.forward, damp * txx - damp * (l + 2 * mu) * dt * vx.forward.dx
                                       - damp * l * dt * (vy.forward.dy + vz.forward.dz))
    u_tyy = Eq(tyy.forward, damp * tyy - damp * (l + 2 * mu) * dt * vy.forward.dy
                                       - damp * l * dt * (vx.forward.dx + vz.forward.dz))
    u_tzz = Eq(tzz.forward, damp * tzz - damp * (l+2*mu)*dt * vz.forward.dz
                                       - damp * l * dt * (vx.forward.dx + vy.forward.dy))
    u_txz = Eq(txz.forward, damp * txz - damp * mu * dt * (vx.forward.dz + vz.forward.dx))
    u_txy = Eq(txy.forward, damp * txy - damp * mu * dt * (vy.forward.dx + vx.forward.dy))
    u_tyz = Eq(tyz.forward, damp * tyz - damp * mu * dt * (vy.forward.dz + vz.forward.dy))

    src_rec_expr = src_rec(vx, vy, vz, txx, tyy, tzz, model, source, receiver)
    return [u_vx, u_vy, u_vz, u_txx, u_tyy, u_tzz, u_txz, u_txy, u_tyz] + src_rec_expr
# ------------------------------------------------------------------------------

def src_rec(vx, vy, vz, txx, tyy, tzz, model, source, receiver):
    """
    Source injection and receiver interpolation
    """
    dt = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, time_range=source.time_range,
                      npoint=source.npoint)
    rec1 = Receiver(name='rec1', grid=model.grid, time_range=receiver.time_range,
                    npoint=receiver.npoint)
    rec2 = Receiver(name='rec2', grid=model.grid, time_range=receiver.time_range,
                    npoint=receiver.npoint)

    # The source injection term
    src_xx = src.inject(field=txx.forward, expr=src * dt, offset=model.nbpml)
    src_zz = src.inject(field=tzz.forward, expr=src * dt, offset=model.nbpml)
    src_expr = src_xx + src_zz
    if model.grid.dim == 3:
        src_yy = src.inject(field=tyy.forward, expr=src * dt, offset=model.nbpml)
        src_expr += src_yy

    # Create interpolation expression for receivers
    rec_term1 = rec1.interpolate(expr=tzz, offset=model.nbpml)
    if model.grid.dim == 2:
        rec_expr = vx.dx + vz.dy
    else:
        rec_expr = vx.dx + vy.dy + vz.dz
    rec_term2 = rec2.interpolate(expr=rec_expr, offset=model.nbpml)

    return src_expr + rec_term1 + rec_term2
# ------------------------------------------------------------------------------

def ForwardOperator(model, source, receiver, space_order=4,
                    save=False, **kwargs):
    """
    Constructor method for the forward modelling operator in an elastic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three buffered
                 indices (last three time steps)
    """

    pde = kernels[model.grid.dim](model, space_order, source.nt if save else None,
                                  source, receiver)

    # Substitute spacing terms to reduce flops
    return Operator(pde, subs=model.spacing_map,
                    name='Forward', **kwargs)


kernels = {3: poroelastic_3d, 2: poroelastic_2d}

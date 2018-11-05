from devito import Eq, Operator, TimeFunction, NODE
from examples.seismic import PointSource, Receiver


def stress_fields(model, save, space_order):
    """
    Create the TimeFunction objects for the stress fields in the elastic formulation
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

    return vx, vy, vz


def elastic_2d(model, space_order, save, source, receiver):
    """
    2D elastic wave equation FD kernel
    """
    vp, vs, rho, damp = model.vp, model.vs, model.rho, model.damp
    s = model.grid.stepping_dim.spacing
    cp2 = vp*vp
    cs2 = vs*vs
    ro = 1/rho

    mu = cs2*rho
    l = rho*(cp2 - 2*cs2)

    # Create symbols for forward wavefield, source and receivers
    vx, vy, vz = particle_velocity_fields(model, save, space_order)
    txx, tyy, tzz, _, txz, _ = stress_fields(model, save, space_order)

    # Stencils
    u_vx = Eq(vx.forward, damp * vx - damp * s * ro * (txx.dx + txz.dy))
    u_vz = Eq(vz.forward, damp * vz - damp * ro * s * (txz.dx + tzz.dy))

    u_txx = Eq(txx.forward, damp * txx - damp * (l + 2 * mu) * s * vx.forward.dx
                                       - damp * l * s * vz.forward.dy)
    u_tzz = Eq(tzz.forward, damp * tzz - damp * (l+2*mu)*s * vz.forward.dy
                                       - damp * l * s * vx.forward.dx)
    u_txz = Eq(txz.forward, damp * txz - damp * mu*s * (vx.forward.dy + vz.forward.dx))

    src_rec_expr = src_rec(vx, vy, vz, txx, tyy, tzz, model, source, receiver)
    return [u_vx, u_vz, u_txx, u_tzz, u_txz] + src_rec_expr


def elastic_3d(model, space_order, save, source, receiver):
    """
    3D elastic wave equation FD kernel
    """
    vp, vs, rho, damp = model.vp, model.vs, model.rho, model.damp
    s = model.grid.stepping_dim.spacing
    cp2 = vp*vp
    cs2 = vs*vs
    ro = 1/rho

    mu = cs2*rho
    l = rho*(cp2 - 2*cs2)

    # Create symbols for forward wavefield, source and receivers
    vx, vy, vz = particle_velocity_fields(model, save, space_order)
    txx, tyy, tzz, txy, txz, tyz = stress_fields(model, save, space_order)

    # Stencils
    u_vx = Eq(vx.forward, damp * vx - damp * s * ro * (txx.dx + txy.dy + txz.dz))
    u_vy = Eq(vy.forward, damp * vy - damp * s * ro * (txy.dx + tyy.dy + tyz.dz))
    u_vz = Eq(vz.forward, damp * vz - damp * s * ro * (txz.dx + tyz.dy + tzz.dz))

    u_txx = Eq(txx.forward, damp * txx - damp * (l + 2 * mu) * s * vx.forward.dx
                                       - damp * l * s * (vy.forward.dy + vz.forward.dz))
    u_tyy = Eq(tyy.forward, damp * tyy - damp * (l + 2 * mu) * s * vy.forward.dy
                                       - damp * l * s * (vx.forward.dx + vz.forward.dz))
    u_tzz = Eq(tzz.forward, damp * tzz - damp * (l+2*mu)*s * vz.forward.dz
                                       - damp * l * s * (vx.forward.dx + vy.forward.dy))
    u_txz = Eq(txz.forward, damp * txz - damp * mu * s * (vx.forward.dz + vz.forward.dx))
    u_txy = Eq(txy.forward, damp * txy - damp * mu * s * (vy.forward.dx + vx.forward.dy))
    u_tyz = Eq(tyz.forward, damp * tyz - damp * mu * s * (vy.forward.dz + vz.forward.dy))

    src_rec_expr = src_rec(vx, vy, vz, txx, tyy, tzz, model, source, receiver)
    return [u_vx, u_vy, u_vz, u_txx, u_tyy, u_tzz, u_txz, u_txy, u_tyz] + src_rec_expr


def src_rec(vx, vy, vz, txx, tyy, tzz, model, source, receiver):
    """
    Source injection and receiver interpolation
    """
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, time_range=source.time_range,
                      npoint=source.npoint)
    rec1 = Receiver(name='rec1', grid=model.grid, time_range=receiver.time_range,
                    npoint=receiver.npoint)
    rec2 = Receiver(name='rec2', grid=model.grid, time_range=receiver.time_range,
                    npoint=receiver.npoint)

    # The source injection term
    src_xx = src.inject(field=txx.forward, expr=src * s)
    src_zz = src.inject(field=tzz.forward, expr=src * s)
    src_expr = src_xx + src_zz
    if model.grid.dim == 3:
        src_yy = src.inject(field=tyy.forward, expr=src * s)
        src_expr += src_yy

    # Create interpolation expression for receivers
    rec_term1 = rec1.interpolate(expr=tzz)
    if model.grid.dim == 2:
        rec_expr = vx.dx + vz.dy
    else:
        rec_expr = vx.dx + vy.dy + vz.dz
    rec_term2 = rec2.interpolate(expr=rec_expr)

    return src_expr + rec_term1 + rec_term2


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


kernels = {3: elastic_3d, 2: elastic_2d}

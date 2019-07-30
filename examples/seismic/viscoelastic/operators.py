import sympy as sp

from devito import Eq, Operator, TimeFunction, NODE
from examples.seismic.elastic import src_rec


def vector_function(name, model, save, space_order):
    """
    Create a vector function such as the particle velocity fields
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_x = x
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name=name+'x', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order)
        vz = TimeFunction(name=name+'z', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order)
        vy = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_x = x
        stagg_y = y
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name=name+'x', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order)
        vy = TimeFunction(name=name+'y', grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order)
        vz = TimeFunction(name=name+'z', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order)

    return vx, vy, vz


def tensor_function(name, model, save, space_order):
    """
    Create a tensor function such as the stress tensor.
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_xx = stagg_zz = NODE
        stagg_xz = (x, z)
        # Create symbols for forward wavefield, source and receivers
        txx = TimeFunction(name=name+'xx', grid=model.grid, staggered=stagg_xx, save=save,
                           time_order=1, space_order=space_order)
        tzz = TimeFunction(name=name+'zz', grid=model.grid, staggered=stagg_zz, save=save,
                           time_order=1, space_order=space_order)
        txz = TimeFunction(name=name+'xz', grid=model.grid, staggered=stagg_xz, save=save,
                           time_order=1, space_order=space_order)
        tyy = txy = tyz = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_xx = stagg_yy = stagg_zz = NODE
        stagg_xz = (x, z)
        stagg_yz = (y, z)
        stagg_xy = (x, y)
        # Create symbols for forward wavefield, source and receivers
        txx = TimeFunction(name=name+'xx', grid=model.grid, staggered=stagg_xx, save=save,
                           time_order=1, space_order=space_order)
        tzz = TimeFunction(name=name+'zz', grid=model.grid, staggered=stagg_zz, save=save,
                           time_order=1, space_order=space_order)
        tyy = TimeFunction(name=name+'yy', grid=model.grid, staggered=stagg_yy, save=save,
                           time_order=1, space_order=space_order)
        txz = TimeFunction(name=name+'xz', grid=model.grid, staggered=stagg_xz, save=save,
                           time_order=1, space_order=space_order)
        txy = TimeFunction(name=name+'xy', grid=model.grid, staggered=stagg_xy, save=save,
                           time_order=1, space_order=space_order)
        tyz = TimeFunction(name=name+'yz', grid=model.grid, staggered=stagg_yz, save=save,
                           time_order=1, space_order=space_order)

    return txx, tyy, tzz, txy, txz, tyz


def viscoelastic_2d(model, space_order, save, geometry):
    """
    2D viscoelastic wave equation FD kernel
    """
    l, qp, mu, qs, ro, damp = \
        model.lam, model.qp, model.mu, model.qs, model.irho, model.damp
    s = model.grid.stepping_dim.spacing

    pi = l + 2*mu

    f0 = geometry._f0
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0
    t_ep = 1./(f0**2*t_s)
    t_es = (1.+f0*qs*t_s)/(f0*qs-f0**2*t_s)

    # Create symbols for forward wavefield, source and receivers
    vx, vy, vz = vector_function('v', model, save, space_order)
    txx, tyy, tzz, _, txz, _ = tensor_function('t', model, save, space_order)
    rxx, ryy, rzz, _, rxz, _ = tensor_function('r', model, save, space_order)

    # Stencils
    u_vx = Eq(vx.forward, damp * vx + damp * s * ro * (txx.dx + txz.dy))
    u_vz = Eq(vz.forward, damp * vz + damp * ro * s * (txz.dx + tzz.dy))

    u_txx = Eq(txx.forward, damp*txx + damp*s*pi*t_ep/t_s*(vx.forward.dx+vz.forward.dy)
               - damp*2.*s*mu*t_es/t_s*(vz.forward.dy) + damp*s*rxx.forward)

    u_tzz = Eq(tzz.forward, damp*tzz + damp*s*pi*t_ep/t_s*(vx.forward.dx+vz.forward.dy)
               - damp*2.*s*mu*t_es/t_s*(vx.forward.dx) + damp*s*rzz.forward)

    u_txz = Eq(txz.forward, damp*txz + damp*s*mu*t_es/t_s*(vx.forward.dy+vz.forward.dx)
               + damp*s*rxz.forward)

    u_rxx = Eq(rxx.forward, damp*rxx
               - damp*s*1./t_s*(rxx+pi*(t_ep/t_s-1)*(vx.forward.dx+vz.forward.dy)
                                - 2*mu*(t_es/t_s-1)*vz.forward.dy))

    u_rzz = Eq(rzz.forward, damp*rzz
               - damp*s*1./t_s*(rzz+pi*(t_ep/t_s-1)*(vx.forward.dx+vz.forward.dy)
                                - 2*mu*(t_es/t_s-1)*vx.forward.dx))

    u_rxz = Eq(rxz.forward, damp*rxz
               - damp*s*1/t_s*(rxz+mu*(t_es/t_s-1)*(vx.forward.dy+vz.forward.dx)))

    src_rec_expr = src_rec(vx, vy, vz, txx, tyy, tzz, model, geometry)
    return [u_vx, u_vz, u_rxx, u_rzz, u_rxz, u_txx, u_tzz, u_txz] + src_rec_expr


def viscoelastic_3d(model, space_order, save, geometry):
    """
    3D viscoelastic wave equation FD kernel
    """
    l, qp, mu, qs, ro, damp = \
        model.lam, model.qp, model.mu, model.qs, model.irho, model.damp
    s = model.grid.stepping_dim.spacing
    pi = l + 2*mu

    f0 = geometry._f0
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0
    t_ep = 1./(f0**2*t_s)
    t_es = (1.+f0*qs*t_s)/(f0*qs-f0**2*t_s)

    # Create symbols for forward wavefield, source and receivers
    vx, vy, vz = vector_function('v', model, save, space_order)
    txx, tyy, tzz, txy, txz, tyz = tensor_function('t', model, save, space_order)
    rxx, ryy, rzz, rxy, rxz, ryz = tensor_function('r', model, save, space_order)

    # Stencils
    u_vx = Eq(vx.forward, damp * vx + damp * s * ro * (txx.dx + txy.dy + txz.dz))
    u_vy = Eq(vy.forward, damp * vy + damp * s * ro * (txy.dx + tyy.dy + tyz.dz))
    u_vz = Eq(vz.forward, damp * vz + damp * s * ro * (txz.dx + tyz.dy + tzz.dz))

    u_txx = Eq(txx.forward, damp*txx
               + damp*s*pi*t_ep/t_s*(vx.forward.dx+vy.forward.dy+vz.forward.dz)
               - damp*2.*s*mu*t_es/t_s*(vy.forward.dy+vz.forward.dz)
               + damp*s*rxx.forward)

    u_tyy = Eq(tyy.forward, damp*tyy
               + damp*s*pi*t_ep/t_s*(vx.forward.dx+vy.forward.dy+vz.forward.dz)
               - damp*2.*s*mu*t_es/t_s*(vx.forward.dx+vz.forward.dz)
               + damp*s*ryy.forward)

    u_tzz = Eq(tzz.forward, damp*tzz
               + damp*s*pi*t_ep/t_s*(vx.forward.dx+vy.forward.dy+vz.forward.dz)
               - damp*2.*s*mu*t_es/t_s*(vx.forward.dx+vy.forward.dy)
               + damp*s*rzz.forward)

    u_txy = Eq(txy.forward, damp*txy
               + damp*s*mu*t_es/t_s*(vx.forward.dy+vy.forward.dx) + damp*s*rxy.forward)

    u_txz = Eq(txz.forward, damp*txz
               + damp*s*mu*t_es/t_s*(vx.forward.dz+vz.forward.dx) + damp*s*rxz.forward)

    u_tyz = Eq(tyz.forward, damp*tyz
               + damp*s*mu*t_es/t_s*(vy.forward.dz+vz.forward.dy) + damp*s*ryz.forward)

    u_rxx = Eq(rxx.forward, damp*rxx
               - damp*s*1./t_s*(rxx+pi*(t_ep/t_s-1)*(vx.forward.dx+vy.forward.dy
                                                     + vz.forward.dz)
                                - 2*mu*(t_es/t_s-1)*(vz.forward.dz+vy.forward.dy)))

    u_ryy = Eq(ryy.forward, damp*ryy
               - damp*s*1./t_s*(ryy+pi*(t_ep/t_s-1)*(vx.forward.dx+vy.forward.dy
                                                     + vz.forward.dz)
                                - 2*mu*(t_es/t_s-1)*(vx.forward.dx+vy.forward.dy)))

    u_rzz = Eq(rzz.forward, damp*rzz
               - damp*s*1./t_s*(rzz+pi*(t_ep/t_s-1)*(vx.forward.dx+vy.forward.dy
                                                     + vz.forward.dz)
                                - 2*mu*(t_es/t_s-1)*(vx.forward.dx+vy.forward.dy)))

    u_rxy = Eq(rxy.forward, damp*rxy
               - damp*s*1/t_s*(rxy+mu*(t_es/t_s-1)*(vx.forward.dy+vy.forward.dx)))

    u_rxz = Eq(rxz.forward, damp*rxz
               - damp*s*1/t_s*(rxz+mu*(t_es/t_s-1)*(vx.forward.dz+vz.forward.dx)))

    u_ryz = Eq(ryz.forward, ryz
               - damp*s*1/t_s*(ryz+mu*(t_es/t_s-1)*(vy.forward.dz+vz.forward.dy)))

    src_rec_expr = src_rec(vx, vy, vz, txx, tyy, tzz, model, geometry)
    return [u_vx, u_vy, u_vz, u_rxx, u_ryy, u_rzz, u_rxz, u_rxy, u_ryz,
            u_txx, u_tyy, u_tzz, u_txz, u_txy, u_tyz] + src_rec_expr


def ForwardOperator(model, geometry, space_order=4, save=False, **kwargs):
    """
    Construct method for the forward modelling operator in an elastic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """
    wave = kernels[model.grid.dim]
    pde = wave(model, space_order, geometry.nt if save else None, geometry)

    # Substitute spacing terms to reduce flops
    return Operator(pde, subs=model.spacing_map,
                    name='Forward', **kwargs)


kernels = {3: viscoelastic_3d, 2: viscoelastic_2d}

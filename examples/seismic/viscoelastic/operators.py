import numpy as np
import sympy as sp

from devito import Eq, Operator
from examples.seismic.elastic import tensor_function, vector_function, src_rec


def viscoelastic_2d(model, space_order, save, geometry):
    """
    2D viscoelastic wave equation FD kernel
    """
    vp, qp, vs, qs, rho, damp = \
        model.vp, model.qp, model.vs, model.qs, model.rho, model.damp
    s = model.grid.stepping_dim.spacing
    cp2 = vp*vp
    cs2 = vs*vs
    ro = 1/rho

    mu = cs2*rho
    l = rho*(cp2 - 2*cs2)
    pi = l + 2*mu

    f0 = geometry._f0
    t_s = (np.sqrt(1.+1./qp**2)-1./qp)/f0
    t_ep = 1./(f0**2*t_s)
    t_es = (1.+f0*qs*t_s)/(f0*qs-f0**2*t_s)

    # Create symbols for forward wavefield, source and receivers
    vx, vy, vz = vector_function('v', model, save, space_order)
    txx, tyy, tzz, _, txz, _ = tensor_function('t', model, save, space_order)
    rxx, ryy, rzz, _, rxz, _ = tensor_function('r', model, save, space_order)

    # Stencils
    u_vx = Eq(vx.forward, damp * vx + damp * s * ro * (txx.dx + txz.dy))
    u_vz = Eq(vz.forward, damp * vz + damp * ro * s * (txz.dx + tzz.dy))

    u_txx = Eq(txx.forward, damp*txx + damp*s*pi*t_ep/t_s*(vx.forward.dx+vz.forward.dz)
               - damp*2.*s*mu*t_es/t_s*(vz.forward.dz) + damp*s*rxx.forward)

    u_tzz = Eq(tzz.forward, damp*tzz + damp*s*pi*t_ep/t_s*(vx.forward.dx+vz.forward.dz)
               - damp*2.*s*mu*t_es/t_s*(vx.forward.dx) + damp*s*rzz.forward)

    u_txz = Eq(txz.forward, damp*txz + damp*s*mu*t_es/t_s*(vx.forward.dz+vz.forward.dx)
               + damp*s*rxz.forward)

    u_rxx = Eq(rxx.forward, damp*rxx
               - damp*s*1./t_s*(rxx+pi*(t_ep/t_s-1)*(vx.forward+vz.forward.dz)
                                - 2*mu*(t_es/t_s-1)*vz.forward.dz))

    u_rzz = Eq(rzz.forward, damp*rzz
               - damp*s*1./t_s*(rzz+pi*(t_ep/t_s-1)*(vx.forward.dx+vz.forward.dz)
                                - 2*mu*(t_es/t_s-1)*vx.forward.dx))

    u_rxz = Eq(rxz.forward, damp*rxz
               - damp*s*1/t_s*(rxz+mu*(t_es/t_s-1)*(vx.forward.dz+vz.forward.dx)))

    src_rec_expr = src_rec(vx, vy, vz, txx, tyy, tzz, rxx, ryy, rzz, model, geometry)
    return [u_vx, u_vz, u_txx, u_tzz, u_txz, u_rxx, u_rzz, u_rxz] + src_rec_expr


def viscoelastic_3d(model, space_order, save, geometry):
    """
    3D viscoelastic wave equation FD kernel
    """
    vp, qp, vs, qs, rho, damp = \
        model.vp, model.qp, model.vs, model.qs, model.rho, model.damp
    s = model.grid.stepping_dim.spacing
    cp2 = vp*vp
    cs2 = vs*vs
    ro = 1/rho

    mu = cs2*rho
    l = rho*(cp2 - 2*cs2)
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
    return [u_vx, u_vy, u_vz, u_txx, u_tyy, u_tzz, u_txz, u_txy, u_tyz,
            u_rxx, u_ryy, u_rzz, u_rxz, u_rxy, u_ryz] + src_rec_expr


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

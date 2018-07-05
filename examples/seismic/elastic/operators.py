
from devito import Eq, Operator, TimeFunction, left, right, staggered_diff
from examples.seismic import PointSource, Receiver


def ForwardOperator(model, source, receiver, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Constructor method for the forward modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    vp, vs, rho, damp = model.vp, model.vs, model.rho, model.damp
    s = model.grid.stepping_dim.spacing
    x, z = model.grid.dimensions
    cp2 = vp*vp
    cs2 = vs*vs
    ro = 1/rho

    mu = cs2*rho
    l = rho*(cp2 - 2*cs2)

    # Create symbols for forward wavefield, source and receivers
    vx = TimeFunction(name='vx', grid=model.grid, staggered=(0, 1, 0),
                      save=source.nt if save else None,
                      time_order=2, space_order=space_order)
    vz = TimeFunction(name='vz', grid=model.grid, staggered=(0, 0, 1),
                      save=source.nt if save else None,
                      time_order=2, space_order=space_order)
    txx = TimeFunction(name='txx', grid=model.grid,
                       save=source.nt if save else None,
                       time_order=2, space_order=space_order)
    tzz = TimeFunction(name='tzz', grid=model.grid,
                       save=source.nt if save else None,
                       time_order=2, space_order=space_order)
    txz = TimeFunction(name='txz', grid=model.grid, staggered=(0, 1, 1),
                       save=source.nt if save else None,
                       time_order=2, space_order=space_order)
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, time_range=source.time_range,
                      npoint=source.npoint)
    rec1 = Receiver(name='rec1', grid=model.grid, time_range=receiver.time_range,
                    npoint=receiver.npoint)
    rec2 = Receiver(name='rec2', grid=model.grid, time_range=receiver.time_range,
                    npoint=receiver.npoint)
    # Stencils
    fd_vx = (staggered_diff(txx, dim=x, order=space_order, stagger=left) +
             staggered_diff(txz, dim=z, order=space_order, stagger=right))
    u_vx = Eq(vx.forward, damp * vx - damp * s * ro * fd_vx)

    fd_vz = (staggered_diff(txz, dim=x, order=space_order, stagger=right) +
             staggered_diff(tzz, dim=z, order=space_order, stagger=left))
    u_vz = Eq(vz.forward, damp * vz - damp * ro * s * fd_vz)

    vxdx = staggered_diff(vx.forward, dim=x, order=space_order, stagger=right)
    vzdz = staggered_diff(vz.forward, dim=z, order=space_order, stagger=right)
    u_txx = Eq(txx.forward, damp * txx - damp * (l + 2 * mu) * s * vxdx
                                       - damp * l * s * vzdz)
    u_tzz = Eq(tzz.forward, damp * tzz - damp * (l+2*mu)*s * vzdz
                                       - damp * l * s * vxdx)

    vxdz = staggered_diff(vx.forward, dim=z, order=space_order, stagger=left)
    vzdx = staggered_diff(vz.forward, dim=x, order=space_order, stagger=left)
    u_txz = Eq(txz.forward, damp * txz - damp * mu*s * (vxdz + vzdx))

    # The source injection term
    src_xx = src.inject(field=txx.forward, expr=src, offset=model.nbpml)
    src_zz = src.inject(field=tzz.forward, expr=src, offset=model.nbpml)

    # Create interpolation expression for receivers
    rec_term1 = rec1.interpolate(expr=txx, offset=model.nbpml)
    rec_term2 = rec2.interpolate(expr=tzz, offset=model.nbpml)
    # Substitute spacing terms to reduce flops
    return Operator([u_vx, u_vz, u_txx, u_tzz, u_txz] + src_xx + src_zz
                    + rec_term1 + rec_term2, subs=model.spacing_map,
                    name='Forward', **kwargs)

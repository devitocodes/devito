from devito import Eq, Operator, VectorTimeFunction, TensorTimeFunction
from devito import div, grad, diag, solve
from examples.seismic import PointSource, Receiver


def src_rec(v, tau, model, geometry, forward=True):
    """
    Source injection and receiver interpolation
    """
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec_vx = Receiver(name='rec_vx', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    rec_vz = Receiver(name='rec_vz', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    if model.grid.dim == 3:
        rec_vy = Receiver(name='rec_vy', grid=model.grid, time_range=geometry.time_axis,
                          npoint=geometry.nrec)
    name = "rec_tau" if forward else "rec"
    rec = Receiver(name="%s" % name, grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    if forward:

        # The source injection term
        src_xx = src.inject(field=tau[0, 0].forward, expr=src * s)
        src_zz = src.inject(field=tau[-1, -1].forward, expr=src * s)
        src_expr = src_xx + src_zz
        if model.grid.dim == 3:
            src_yy = src.inject(field=tau[1, 1].forward, expr=src * s)
            src_expr += src_yy
        # Create interpolation expression for receivers
        rec_term_vx = rec_vx.interpolate(expr=v[0])
        rec_term_vz = rec_vz.interpolate(expr=v[-1])
        expr = tau[0, 0] + tau[-1, -1]
        rec_expr = rec_term_vx + rec_term_vz
        if model.grid.dim == 3:
            expr += tau[1, 1]
            rec_term_vy = rec_vy.interpolate(expr=v[1])
            rec_expr += rec_term_vy
        rec_term_tau = rec.interpolate(expr=expr)
        rec_expr += rec_term_tau

    else:
        # Construct expression to inject receiver values
        rec_xx = rec.inject(field=tau[0, 0].backward, expr=rec*s)
        rec_zz = rec.inject(field=tau[-1, -1].backward, expr=rec*s)
        rec_expr = rec_xx + rec_zz
        expr = tau[0, 0] + tau[-1, -1]
        if model.grid.dim == 3:
            rec_expr += rec.inject(field=tau[1, 1].backward, expr=rec*s)
            expr += tau[1, 1]
        # Create interpolation expression for the adjoint-source
        src_expr = src.interpolate(expr=expr)

    return src_expr, rec_expr


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

    v = VectorTimeFunction(name='v', grid=model.grid,
                           save=geometry.nt if save else None,
                           space_order=space_order, time_order=1)
    tau = TensorTimeFunction(name='tau', grid=model.grid,
                             save=geometry.nt if save else None,
                             space_order=space_order, time_order=1)

    lam, mu, b = model.lam, model.mu, model.b

    # Particle velocity
    eq_v = v.dt - b * div(tau)
    # Stress
    e = (grad(v.forward) + grad(v.forward).T)
    eq_tau = tau.dt - lam * diag(div(v.forward)) - mu * e

    u_v = Eq(v.forward, model.damp * solve(eq_v, v.forward))
    u_t = Eq(tau.forward, model.damp * solve(eq_tau, tau.forward))

    srcrec = src_rec(v, tau, model, geometry)
    op = Operator([u_v] + [u_t] + srcrec, subs=model.spacing_map, name="ForwardElastic",
                  **kwargs)
    # Substitute spacing terms to reduce flops
    return op

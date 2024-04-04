from devito import Eq, Operator, VectorTimeFunction, TensorTimeFunction
from devito import div, grad, diag, solve


def src_rec(v, tau, model, geometry):
    """
    Source injection and receiver interpolation
    """
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = geometry.src
    rec1 = geometry.new_rec(name="rec1")
    rec2 = geometry.new_rec(name="rec2")

    # The source injection term
    src_expr = src.inject(tau.forward.diagonal(), expr=src * s)

    # Create interpolation expression for receivers
    rec_term1 = rec1.interpolate(expr=tau[-1, -1])
    rec_term2 = rec2.interpolate(expr=div(v))

    return src_expr + rec_term1 + rec_term2


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
    e = (grad(v.forward) + grad(v.forward).transpose(inner=False))
    eq_tau = tau.dt - lam * diag(div(v.forward)) - mu * e

    u_v = Eq(v.forward, model.damp * solve(eq_v, v.forward))
    u_t = Eq(tau.forward, model.damp * solve(eq_tau, tau.forward))

    srcrec = src_rec(v, tau, model, geometry)
    op = Operator([u_v] + [u_t] + srcrec, subs=model.spacing_map, name="ForwardElastic",
                  **kwargs)
    # Substitute spacing terms to reduce flops
    return op

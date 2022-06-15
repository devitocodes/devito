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


def elastic_stencil(model, v, tau, forward=True):

    lam, mu, b, damp = model.lam, model.mu, model.b, model.damp

    rho = 1. / b

    if forward:

        # Particle velocity
        eq_v = v.dt - b * div(tau)
        # Stress
        e = (grad(v.forward) + grad(v.forward).T)
        eq_tau = tau.dt - lam * diag(div(v.forward)) - mu * e

        u_v = Eq(v.forward, damp * solve(eq_v, v.forward))
        u_t = Eq(tau.forward, damp * solve(eq_tau, tau.forward))

        return [u_v, u_t]

    else:

        """
        Implementation of the elastic wave-equation from:
        1 - Feng and Schuster (2017): Elastic least-squares reverse time migration
        https://doi.org/10.1190/geo2016-0254.1

        """

        lpmu = lam + 2. * mu

        pde_vx = rho * v[0].dt.T - (lpmu * tau[0, 0]).dx - (lam * tau[1, 1]).dx - \
            (mu * tau[0, 1]).dy
        pde_vy = rho * v[1].dt.T - (lpmu * tau[1, 1]).dy - (lam * tau[0, 0]).dy - \
            (mu * tau[0, 1]).dx
        pde_tauxx = tau[0, 0].dt.T - v[0].backward.dx
        pde_tauyy = tau[1, 1].dt.T - v[1].backward.dy
        pde_tauxy = tau[0, 1].dt.T - v[0].backward.dy - v[1].backward.dx

        if model.grid.dim == 3:
            pde_vz = rho * v[2].dt.T - (lpmu * tau[2, 2]).dz - (lam * tau[0, 0]).dz - \
                (lam * tau[1, 1]).dz - (mu * tau[0, 2]).dx - (mu * tau[1, 2]).dy
            u_vz = Eq(v[2].backward, damp * solve(pde_vz, v[2].backward))

            pde_tauzz = tau[2, 2].dt.T - v[2].backward.dz
            u_tauzz = Eq(tau[2, 2].backward, damp * solve(pde_tauzz, tau[2, 2].backward))

            pde_vx += -(lam * tau[2, 2]).dx - (mu * tau[0, 2]).dz
            pde_vy += -(lam * tau[2, 2]).dy - (mu * tau[1, 2]).dz

            pde_tauxz = tau[0, 2].dt.T - v[0].backward.dz - v[2].backward.dx
            u_tauxz = Eq(tau[0, 2].backward, damp * solve(pde_tauxz, tau[0, 2].backward))

            pde_tauyz = tau[1, 2].dt.T - v[1].backward.dz - v[2].backward.dy
            u_tauyz = Eq(tau[1, 2].backward, damp * solve(pde_tauyz, tau[1, 2].backward))

        u_vx = Eq(v[0].backward, damp * solve(pde_vx, v[0].backward))
        u_vy = Eq(v[1].backward, damp * solve(pde_vy, v[1].backward))
        u_tauxx = Eq(tau[0, 0].backward, damp * solve(pde_tauxx, tau[0, 0].backward))
        u_tauyy = Eq(tau[1, 1].backward, damp * solve(pde_tauyy, tau[1, 1].backward))
        u_tauxy = Eq(tau[0, 1].backward, damp * solve(pde_tauxy, tau[0, 1].backward))

        if model.grid.dim == 3:
            kernels = \
                u_vx, u_vy, u_tauxx, u_tauyy, u_tauxy, u_vz, u_tauzz, u_tauxz, u_tauyz
        else:
            kernels = u_vx, u_vy, u_tauxx, u_tauyy, u_tauxy

        return [kernels]


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

    eqn = elastic_stencil(model, v, tau)

    src_expr, rec_expr = src_rec(v, tau, model, geometry)

    op = Operator(eqn + src_expr + rec_expr, subs=model.spacing_map,
                  name="ForwardElastic", **kwargs)
    # Substitute spacing terms to reduce flops
    return op


def AdjointOperator(model, geometry, space_order=4, **kwargs):
    """
    Construct an adjoint modelling operator in a viscoacoustic medium.
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

    u = VectorTimeFunction(name='u', grid=model.grid, space_order=space_order,
                           time_order=1)
    sig = TensorTimeFunction(name='sig', grid=model.grid, space_order=space_order,
                             time_order=1)

    eqn = elastic_stencil(model, u, sig, forward=False)

    src_expr, rec_expr = src_rec(u, sig, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_expr + rec_expr, subs=model.spacing_map,
                    name='AdjointElastic', **kwargs)

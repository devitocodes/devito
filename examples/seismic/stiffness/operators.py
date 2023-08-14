from devito import (Eq, Operator, VectorTimeFunction, TensorTimeFunction,
                    Function, TimeFunction)
from devito import solve
from examples.seismic import PointSource, Receiver
from examples.seismic.stiffness.utils import D, S, vec, C_Matrix, gather


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
        src_xx = src.inject(field=tau[0].forward, expr=src * s)
        src_zz = src.inject(field=tau[1].forward, expr=src * s)
        src_expr = src_xx + src_zz
        if model.grid.dim == 3:
            src_yy = src.inject(field=tau[2].forward, expr=src * s)
            src_expr += src_yy
        # Create interpolation expression for receivers
        rec_term_vx = rec_vx.interpolate(expr=v[0])
        rec_term_vz = rec_vz.interpolate(expr=v[1])
        expr = tau[0] + tau[1]
        rec_expr = rec_term_vx + rec_term_vz
        if model.grid.dim == 3:
            expr += tau[2]
            rec_term_vy = rec_vy.interpolate(expr=v[2])
            rec_expr += rec_term_vy
        rec_term_tau = rec.interpolate(expr=expr)
        rec_expr += rec_term_tau

    else:
        # Construct expression to inject receiver values
        rec_xx = rec.inject(field=tau[0].backward, expr=rec*s)
        rec_zz = rec.inject(field=tau[1].backward, expr=rec*s)
        rec_expr = rec_xx + rec_zz
        expr = tau[0] + tau[1]
        if model.grid.dim == 3:
            rec_expr += rec.inject(field=tau[2].backward, expr=rec*s)
            expr += tau[2]
        # Create interpolation expression for the adjoint-source
        src_expr = src.interpolate(expr=expr)

    return src_expr, rec_expr


def elastic_stencil(model, v, tau, forward=True, par='lam-mu'):

    damp = model.damp
    b = model.b

    rho = 1. / b

    C = C_Matrix(model, par)

    tau = vec(tau)
    if forward:

        pde_v = rho * v.dt - D(tau)
        u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

        pde_tau = tau.dt - C * S(v.forward)
        u_t = Eq(tau.forward, damp * solve(pde_tau, tau.forward))

        return [u_v, u_t]

    else:

        """
        Implementation of the elastic wave-equation from:
        1 - Feng and Schuster (2017): Elastic least-squares reverse time migration
        https://doi.org/10.1190/geo2016-0254.1
        """

        pde_v = rho * v.dtl - D(C.T*tau)
        u_v = Eq(v.backward, damp * solve(pde_v, v.backward))

        pde_tau = -tau.dtl + S(v.backward)
        u_t = Eq(tau.backward, damp * solve(pde_tau, tau.backward))

        return [u_v, u_t]


def EqsLamMu(model, sig, u, v, grad_lam, grad_mu, grad_rho, C, space_order=8):
    hl = TimeFunction(name='hl', grid=model.grid, space_order=space_order,
                      time_order=1)
    hm = TimeFunction(name='hm', grid=model.grid, space_order=space_order,
                      time_order=1)
    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)

    Wl = gather(0, C.dlam * S(v))
    Wm = gather(0, C.dmu * S(v))
    Wr = gather(v.dt, 0)

    W2 = gather(u, sig)

    wl_update = Eq(hl, Wl.T * W2)
    gradient_lam = Eq(grad_lam, grad_lam + hl)

    wm_update = Eq(hm, Wm.T * W2)
    gradient_mu = Eq(grad_mu, grad_mu + hm)

    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)

    return [wl_update, gradient_lam, wm_update, gradient_mu, wr_update, gradient_rho]


def EqsVpVsRho(model, sig, u, v, grad_vp, grad_vs, grad_rho, C, space_order=8):
    hvp = TimeFunction(name='hvp', grid=model.grid, space_order=space_order,
                       time_order=1)
    hvs = TimeFunction(name='hvs', grid=model.grid, space_order=space_order,
                       time_order=1)
    hr = TimeFunction(name='hr', grid=model.grid, space_order=space_order,
                      time_order=1)

    Wvp = gather(0, C.dvp * S(v))
    Wvs = gather(0, C.dvs * S(v))
    Wr = gather(v.dt, - C.drho * S(v))

    W2 = gather(u, sig)

    wvp_update = Eq(hvp, Wvp.T * W2)
    gradient_lam = Eq(grad_vp, grad_vp + hvp)

    wvs_update = Eq(hvs, Wvs.T * W2)
    gradient_mu = Eq(grad_vs, grad_vs + hvs)

    wr_update = Eq(hr, Wr.T * W2)
    gradient_rho = Eq(grad_rho, grad_rho - hr)

    return [wvp_update, gradient_lam, wvs_update, gradient_mu, wr_update, gradient_rho]


def ForwardOperator(model, geometry, space_order=4, save=False, par='lam-mu', **kwargs):
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

    eqn = elastic_stencil(model, v, tau, par=par)

    src_expr, rec_expr = src_rec(v, tau, model, geometry)

    op = Operator(eqn + src_expr + rec_expr, subs=model.spacing_map,
                  name="ForwardIsoElastic", **kwargs)
    # Substitute spacing terms to reduce flops
    return op


def AdjointOperator(model, geometry, space_order=4, par='lam-mu', **kwargs):
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

    eqn = elastic_stencil(model, u, sig, forward=False, par=par)

    src_expr, rec_expr = src_rec(u, sig, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_expr + rec_expr, subs=model.spacing_map,
                    name='AdjointIsoElastic', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True, par='lam-mu', **kwargs):
    """
    Construct a gradient operator in an elastic media.
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
    # Gradient symbol and wavefield symbols
    grad1 = Function(name='grad1', grid=model.grid)
    grad2 = Function(name='grad2', grid=model.grid)
    grad3 = Function(name='grad3', grid=model.grid)

    v = VectorTimeFunction(name='v', grid=model.grid,
                           save=geometry.nt if save else None,
                           space_order=space_order, time_order=1)
    u = VectorTimeFunction(name='u', grid=model.grid, space_order=space_order,
                           time_order=1)
    sig = TensorTimeFunction(name='sig', grid=model.grid, space_order=space_order,
                             time_order=1)
    rec_vx = Receiver(name='rec_vx', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    rec_vz = Receiver(name='rec_vz', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nrec)
    if model.grid.dim == 3:
        rec_vy = Receiver(name='rec_vy', grid=model.grid, time_range=geometry.time_axis,
                          npoint=geometry.nrec)

    s = model.grid.time_dim.spacing
    b = model.b

    C = C_Matrix(model, par)

    eqn = elastic_stencil(model, u, sig, forward=False, par=par)
    sig = vec(sig)

    kernel = kernels[par]
    gradient_update = kernel(model, sig, u, v, grad1, grad2,
                             grad3, C, space_order=space_order)

    # Construct expression to inject receiver values
    rec_term_vx = rec_vx.inject(field=u[0].backward, expr=s*rec_vx*b)
    rec_term_vz = rec_vz.inject(field=u[-1].backward, expr=s*rec_vz*b)
    rec_expr = rec_term_vx + rec_term_vz
    if model.grid.dim == 3:
        rec_expr += rec_vy.inject(field=u[1].backward, expr=s*rec_vy*b)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + rec_expr + gradient_update, subs=model.spacing_map,
                    name='GradientElastic', **kwargs)


kernels = {'lam-mu': EqsLamMu, 'vp-vs-rho': EqsVpVsRho}

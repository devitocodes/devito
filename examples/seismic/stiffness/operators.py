from devito import Eq, Operator, VectorTimeFunction, TensorTimeFunction
from devito import solve
from examples.seismic import PointSource, Receiver
from examples.seismic.stiffness.utils import D, S, vec, matriz_init


def iso_elastic_tensor(model):
    def subs3D(lmbda, mu):
        return {'C11': lmbda + (2*mu),
                'C22': lmbda + (2*mu),
                'C33': lmbda + (2*mu),
                'C44': mu,
                'C55': mu,
                'C66': mu,
                'C12': lmbda,
                'C13': lmbda,
                'C23': lmbda}

    def subs2D(lmbda, mu):
        return {'C11': lmbda + (2*mu),
                'C22': lmbda + (2*mu),
                'C33': mu,
                'C12': lmbda}

    matriz = matriz_init(model)
    lmbda = model.lam
    mu = model.mu

    subs = subs3D(lmbda, mu) if model.dim == 3 else subs2D(lmbda, mu)
    return matriz.subs(subs)


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


def elastic_stencil(model, v, tau, forward=True):

    damp = model.damp
    b = model.b

    rho = 1. / b

    C = iso_elastic_tensor(model)
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
                  name="ForwardIsoElastic", **kwargs)
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
                    name='AdjointIsoElastic', **kwargs)

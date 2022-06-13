import sympy as sp

from devito import (Eq, Operator, VectorTimeFunction, TensorTimeFunction,
                    div, grad, diag, solve)
from examples.seismic.elastic import src_rec


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
    l, qp, mu, qs, b, damp = \
        model.lam, model.qp, model.mu, model.qs, model.b, model.damp

    f0 = geometry._f0
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0
    t_ep = 1./(f0**2*t_s)
    t_es = (1.+f0*qs*t_s)/(f0*qs-f0**2*t_s)

    # Create symbols for forward wavefield, source and receivers
    # Velocity:
    v = VectorTimeFunction(name="v", grid=model.grid,
                           save=geometry.nt if save else None,
                           time_order=1, space_order=space_order)
    # Stress:
    tau = TensorTimeFunction(name='t', grid=model.grid,
                             save=geometry.nt if save else None,
                             space_order=space_order, time_order=1)
    # Memory variable:
    r = TensorTimeFunction(name='r', grid=model.grid,
                           save=geometry.nt if save else None,
                           space_order=space_order, time_order=1)

    # Particle velocity
    pde_v = v.dt - b * div(tau)
    u_v = Eq(v.forward, model.damp * solve(pde_v, v.forward))
    # Strain
    e = grad(v.forward) + grad(v.forward).T

    # Stress equations
    pde_tau = tau.dt - r.forward - l * t_ep / t_s * diag(div(v.forward)) - \
        mu * t_es / t_s * e
    u_t = Eq(tau.forward, model.damp * solve(pde_tau, tau.forward))

    # Memory variable equations:
    pde_r = r.dt + 1 / t_s * (r + l * (t_ep/t_s-1) * diag(div(v.forward)) +
                              mu * (t_es / t_s - 1) * e)
    u_r = Eq(r.forward, damp * solve(pde_r, r.forward))
    # Point source
    src_rec_expr = src_rec(v, tau, model, geometry)

    # Substitute spacing terms to reduce flops
    return Operator([u_v, u_r, u_t] + src_rec_expr, subs=model.spacing_map,
                    name='Forward', **kwargs)

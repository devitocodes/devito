import sympy as sp
import numpy as np

from devito import (Eq, Operator, VectorTimeFunction, TimeFunction, NODE,
                    div, grad, diag)
from examples.seismic import PointSource, Receiver

def viscoacoustic_blanch_symes(v, r, p, vp, irho, rho, t_s, tt, s, damp, **kwargs):
    """
    Stencil created from from Blanch and Symes (1995) / Dutta and Schuster (2014)
    viscoacoustic wave equation.

    Parameters
    ----------
    v : VectorTimeFunction
        The computed particle velocity.
    r : TimeFunction
        Memory variable.
    p : TimeFunction
        The computed solution.
    vp : Function or float
        The time-constant velocity.
    irho : Function
        The time-constant inverse density.
    rho : Function
        The time-constant density.
    t_s : Float
        The relaxation parameter.
    tt : Float
        The relaxation parameter.
    s : float or Scalar
        The time dimension spacing.
    damp : Function
        The damping field for absorbing boundary condition.
    """
    # Bulk modulus
    bm = rho * (vp * vp)

    # Define PDE
    pde_v = v - s * irho * grad(p)

    u_v = Eq(v.forward, damp * pde_v)

    pde_r = r - s * (1. / t_s) * r - s * (1. / t_s) * tt * bm * div(v.forward)

    u_r = Eq(r.forward, damp * pde_r)

    pde_p = p - s * bm * (tt + 1.) * div(v.forward) - s * r.forward

    u_p = Eq(p.forward, damp * pde_p)

    return [u_v, u_r, u_p]

def viscoacoustic_ren(v, p, vp, irho, rho, qp, f0, s, damp, **kwargs):
    """
    Stencil created from Ren et al. (2014) viscoacoustic wave equation

    Parameters
    ----------
    v : VectorTimeFunction
        The computed particle velocity.
    p : TimeFunction
        The computed solution.
    vp : Function or float
        The time-constant velocity.
    irho : Function
        The time-constant inverse density.
    rho : Function
        The time-constant density.
    qp : Function
        The P-wave quality factor.
    f0: float or Scalar
        The dominant frequency
    s : float or Scalar
        The time dimension spacing.
    damp : Function
        The damping field for absorbing boundary condition.
    """
    # Angular frequency
    w = 2. * np.pi * f0

    # Define PDE
    pde_v = v - s * irho * grad(p)

    u_v = Eq(v.forward, damp * pde_v)

    pde_u = p - s * vp * vp * rho * div(v.forward) + s * ((vp * vp * rho) / \
            (w * qp)) * div(irho * grad(p))

    u_p = Eq(p.forward, damp * pde_u)

    return [u_v, u_p]

def viscoacoustic_deng_mcmechan(v, p, vp, irho, rho, qp, f0, s, damp, **kwargs):
    """
    Stencil created from Deng and McMechan (2007) viscoacoustic wave equation

    Parameters
    ----------
    v : VectorTimeFunction
        The computed particle velocity.
    p : TimeFunction
        The computed solution.
    vp : Function or float
        The time-constant velocity.
    irho : Function
        The time-constant inverse density.
    rho : Function
        The time-constant density.
    qp : Function
        The P-wave quality factor.
    f0: float or Scalar
        The dominant frequency
    s : float or Scalar
        The time dimension spacing.
    damp : Function
        The damping field for absorbing boundary condition.
    """
    # Angular frequency
    w = 2. * np.pi * f0
    # Define PDE
    pde_v = v - s * irho * grad(p)

    u_v = Eq(v.forward, damp * pde_v)

    pde_p = p - s * vp * vp * rho * div(v.forward) - s * (w / qp) * p

    u_p = Eq(p.forward, damp * pde_p)

    return [u_v, u_p]

def ForwardOperator(model, geometry, space_order=4, equation=1, save=False, **kwargs):
    """
    Construct method for the forward modelling operator in an viscoacoustic media.

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
    equation : selects a visco-acoustic equation from the options below:
                1 - Blanch and Symes (1995) / Dutta and Schuster (2014) viscoacoustic equation
                2 - Ren et al. (2014) viscoacoustic equation
                3 - Deng and McMechan (2007) viscoacoustic equation
                Defaults to 1.
    """
    qp, rho, irho, vp, damp = \
        model.qp, model.rho, model.irho, model.vp, model.damp
    s = model.grid.stepping_dim.spacing

    f0 = geometry._f0
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0
    t_ep = 1./(f0**2*t_s)
    tt = (t_ep/t_s)-1.

    # Create symbols for forward wavefield, particle velocity, source and receivers
    # Velocity:
    v = VectorTimeFunction(name="v", grid=model.grid,
                           save=geometry.nt if save else None,
                           time_order=1, space_order=space_order)

    r = TimeFunction(name="r", grid=model.grid, staggered=NODE,
                     save=geometry.nt if save else None,
                     time_order=1, space_order=space_order)

    p = TimeFunction(name="p", grid=model.grid, staggered=NODE,
                     save=geometry.nt if save else None,
                     time_order=1, space_order=space_order)

    src = PointSource(name="src", grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)

    rec = Receiver(name="rec", grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)

    if equation == 1:
        # Implements PDEs from Blanch and Symes (1995) Dutta and Schuster (2014)
        # viscoacoustic equation
        eqn = viscoacoustic_blanch_symes(v, r, p, vp, irho, rho, t_s, tt, s, damp)

    elif equation == 2:
        # Implements PDEs from Ren et al. (2014) viscoacoustic equation
        eqn = viscoacoustic_ren(v, p, vp, irho, rho, qp, f0, s, damp)

    elif equation == 3:
        # Implements PDEs from Deng and McMechan (2007) viscoacoustic equation
        eqn = viscoacoustic_deng_mcmechan(v, p, vp, irho, rho, qp, f0, s, damp)

    else:
        # Implements PDEs from Blanch and Symes (1995) Dutta and Schuster (2014)
        # viscoacoustic equation
        eqn = viscoacoustic_blanch_symes(v, r, p, vp, irho, rho, t_s, tt, s, damp)

    # The source injection term
    src_term = src.inject(field=p.forward, expr=src * s)
    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=p.forward)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)
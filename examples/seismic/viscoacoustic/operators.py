import sympy as sp
import numpy as np

from devito import (Eq, Operator, VectorTimeFunction, TimeFunction, NODE,
                    div, grad)
from examples.seismic import PointSource, Receiver


def blanch_symes(model, geometry, v, p, **kwargs):
    """
    Stencil created from from Blanch and Symes (1995) / Dutta and Schuster (2014)
    viscoacoustic wave equation.

    https://library.seg.org/doi/pdf/10.1190/1.1822695
    https://library.seg.org/doi/pdf/10.1190/geo2013-0414.1

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    space_order = kwargs.get('space_order')
    save = kwargs.get('save')

    s = model.grid.stepping_dim.spacing
    irho = model.irho
    vp = model.vp
    damp = model.damp
    qp = model.qp
    f0 = geometry._f0

    # The stress relaxation parameter
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0

    # The strain relaxation parameter
    t_ep = 1./(f0**2*t_s)

    # The relaxation time
    tt = (t_ep/t_s)-1.

    # Density
    rho = 1. / irho

    # Bulk modulus
    bm = rho * (vp * vp)

    # Memory variable.
    r = TimeFunction(name="r", grid=model.grid, staggered=NODE,
                     save=geometry.nt if save else None,
                     time_order=1, space_order=space_order)

    # Define PDE
    pde_v = v - s * irho * grad(p)

    u_v = Eq(v.forward, damp * pde_v)

    pde_r = r - s * (1. / t_s) * r - s * (1. / t_s) * tt * bm * div(v.forward)

    u_r = Eq(r.forward, damp * pde_r)

    pde_p = p - s * bm * (tt + 1.) * div(v.forward) - s * r.forward

    u_p = Eq(p.forward, damp * pde_p)

    return [u_v, u_r, u_p]


def ren(model, geometry, v, p, **kwargs):
    """
    Stencil created from Ren et al. (2014) viscoacoustic wave equation.

    https://academic.oup.com/gji/article/197/2/948/616510

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    irho = model.irho
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w = 2. * np.pi * f0

    # Density
    rho = 1. / irho

    # Define PDE
    pde_v = v - s * irho * grad(p)

    u_v = Eq(v.forward, damp * pde_v)

    pde_u = p - s * vp * vp * rho * div(v.forward) + \
        s * ((vp * vp * rho) / (w * qp)) * div(irho * grad(p))

    u_p = Eq(p.forward, damp * pde_u)

    return [u_v, u_p]


def deng_mcmechan(model, geometry, v, p, **kwargs):
    """
    Stencil created from Deng and McMechan (2007) viscoacoustic wave equation.

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    irho = model.irho
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w = 2. * np.pi * f0

    # Density
    rho = 1. / irho

    # Define PDE
    pde_v = v - s * irho * grad(p)

    u_v = Eq(v.forward, damp * pde_v)

    pde_p = p - s * vp * vp * rho * div(v.forward) - s * (w / qp) * p

    u_p = Eq(p.forward, damp * pde_p)

    return [u_v, u_p]


def ForwardOperator(model, geometry, space_order=4, kernel='blanch_symes', save=False,
                    **kwargs):
    """
    Construct method for the forward modelling operator in a viscoacoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : selects a visco-acoustic equation from the options below:
        blanch_symes - Blanch and Symes (1995) / Dutta and Schuster (2014)
        viscoacoustic equation
        ren - Ren et al. (2014) viscoacoustic equation
        deng_mcmechan - Deng and McMechan (2007) viscoacoustic equation
        Defaults to blanch_symes.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """
    s = model.grid.stepping_dim.spacing

    # Create symbols for forward wavefield, particle velocity, source and receivers
    # Velocity:
    v = VectorTimeFunction(name="v", grid=model.grid,
                           save=geometry.nt if save else None,
                           time_order=1, space_order=space_order)

    p = TimeFunction(name="p", grid=model.grid, staggered=NODE,
                     save=geometry.nt if save else None,
                     time_order=1, space_order=space_order)

    src = PointSource(name="src", grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)

    rec = Receiver(name="rec", grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, v, p, space_order=space_order, save=save)

    # The source injection term
    src_term = src.inject(field=p.forward, expr=src * s)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=p)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


kernels = {'blanch_symes': blanch_symes, 'ren': ren, 'deng_mcmechan': deng_mcmechan}

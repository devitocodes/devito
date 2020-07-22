import sympy as sp
import numpy as np

from devito import (Eq, Operator, VectorTimeFunction, TimeFunction, NODE,
                    div, grad)
from examples.seismic import PointSource, Receiver


def sls_1st_stencil(model, geometry, v, p, **kwargs):
    """
    Stencil created from Blanch and Symes (1995) / Dutta and Schuster (2014)
    viscoacoustic wave equation.

    https://library.seg.org/doi/pdf/10.1190/1.1822695
    https://library.seg.org/doi/pdf/10.1190/geo2013-0414.1

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    save = kwargs.get('save', False)
    s = model.grid.stepping_dim.spacing
    b = model.b
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
    rho = 1. / b

    # Bulk modulus
    bm = rho * (vp * vp)

    # Memory variable.
    r = TimeFunction(name="r", grid=model.grid, staggered=NODE,
                     save=geometry.nt if save else None,
                     time_order=1)

    # Define PDE
    pde_v = v - s * b * grad(p)
    u_v = Eq(v.forward, damp * pde_v)

    pde_r = r - s * (1. / t_s) * r - s * (1. / t_s) * tt * bm * div(v.forward)
    u_r = Eq(r.forward, damp * pde_r)

    pde_p = p - s * bm * (tt + 1.) * div(v.forward) - s * r.forward
    u_p = Eq(p.forward, damp * pde_p)

    return [u_v, u_r, u_p]


def sls_2nd_stencil(model, geometry, v, p, **kwargs):
    """
    Stencil created from Bai (2014) viscoacoustic wave equation.

    https://library.seg.org/doi/10.1190/geo2013-0030.1

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward')
    space_order = p.space_order
    s = model.grid.stepping_dim.spacing
    b = model.b
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
    rho = 1. / b

    r = TimeFunction(name="r", grid=model.grid, time_order=2, space_order=space_order,
                     staggered=NODE)

    if forward:

        u_v = Eq(v, b * grad(p))

        pde_r = r + s * (tt / t_s) * rho * div(v) - s * (1. / t_s) * r
        u_r = Eq(r.forward, damp * pde_r)

        pde_p = 2. * p - damp * p.backward + s * s * vp * vp * (1. + tt) * rho * \
            div(v) - s * s * vp * vp * r.forward
        u_p = Eq(p.forward, damp * pde_p)

        return [u_v, u_r, u_p]

    else:

        # Auxiliary functions
        w = VectorTimeFunction(name="w", grid=model.grid, time_order=2,
                               space_order=space_order)
        h = TimeFunction(name="h", grid=model.grid, time_order=2, space_order=space_order,
                         staggered=NODE)
        g = TimeFunction(name="g", grid=model.grid, time_order=2, space_order=space_order,
                         staggered=NODE)

        pde_r = r + s * (tt / t_s) * p - s * (1. / t_s) * r
        u_r = Eq(r.backward, damp * pde_r)

        u_h = Eq(h, (1. + tt) * rho * p)

        u_v = Eq(v, b * grad(h))

        u_g = Eq(g, rho * r.backward)

        u_w = Eq(w, b * grad(g))

        pde_p = 2. * p - damp * p.forward + s * s * vp * vp * div(v) - \
            s * s * vp * vp * div(w)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_r, u_h, u_v, u_g, u_w, u_p]


def ren_1st_stencil(model, geometry, v, p, **kwargs):
    """
    Stencil created from Ren et al. (2014) viscoacoustic wave equation.

    https://academic.oup.com/gji/article/197/2/948/616510

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    # Define PDE
    pde_v = v - s * b * grad(p)
    u_v = Eq(v.forward, damp * pde_v)

    pde_p = p - s * vp * vp * rho * div(v.forward) + \
        s * ((vp * vp * rho) / (w0 * qp)) * div(b * grad(p))
    u_p = Eq(p.forward, damp * pde_p)

    return [u_v, u_p]


def ren_2nd_stencil(model, geometry, v, p, **kwargs):
    """
    Stencil created from Ren et al. (2014) viscoacoustic wave equation.

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward')
    space_order = p.space_order

    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    eta = (vp * vp) / (w0 * qp)

    # Bulk modulus
    bm = rho * (vp * vp)

    if forward:

        # Auxiliary functions
        h = TimeFunction(name="h", grid=model.grid, time_order=2, space_order=space_order,
                         staggered=NODE)
        w = VectorTimeFunction(name="w", grid=model.grid, time_order=2,
                               space_order=space_order)

        u_h = Eq(h, (p - p.backward) / s)

        u_v = Eq(v, b * grad(p))

        u_w = Eq(w, b * grad(h))

        pde_p = 2. * p - damp * p.backward + s * s * bm * div(v) + \
            s * s * eta * rho * div(w)

        u_p = Eq(p.forward, damp * pde_p)

        return [u_h, u_v, u_w, u_p]

    else:

        # Auxiliary functions
        w = VectorTimeFunction(name="w", grid=model.grid, time_order=2,
                               space_order=space_order)
        h = TimeFunction(name="h", grid=model.grid, time_order=2, space_order=space_order,
                         staggered=NODE)
        g = TimeFunction(name="g", grid=model.grid, time_order=2, space_order=space_order,
                         staggered=NODE)

        u_h = Eq(h, bm * p)

        u_w = Eq(w, b * grad(h))

        u_g = Eq(g, ((p.forward - p) / s) * rho * eta)

        u_v = Eq(v, b * grad(g))

        pde_p = 2. * p - damp * p.forward + s * s * div(w) - s * s * div(v)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_h, u_w, u_g, u_v, u_p]


def deng_1st_stencil(model, geometry, v, p, **kwargs):
    """
    Stencil created from Deng and McMechan (2007) viscoacoustic wave equation.

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    # Define PDE
    pde_v = v - s * b * grad(p)
    u_v = Eq(v.forward, damp * pde_v)

    pde_p = p - s * vp * vp * rho * div(v.forward) - s * (w0 / qp) * p
    u_p = Eq(p.forward, damp * pde_p)

    return [u_v, u_p]


def deng_2nd_stencil(model, geometry, v, p, **kwargs):
    """
    Stencil created from Deng and McMechan (2007) viscoacoustic wave equation.

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward')
    space_order = p.space_order

    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    bm = rho * (vp * vp)

    if forward:

        # Auxiliary functions
        w = VectorTimeFunction(name="w", grid=model.grid, time_order=2,
                               space_order=space_order)
        h = TimeFunction(name="h", grid=model.grid, time_order=2, space_order=space_order,
                         staggered=NODE)

        u_h = Eq(h, (p - p.backward) / s)

        u_v = Eq(v, b * grad(p))

        u_w = Eq(w, b * grad(h))

        pde_p = 2. * p - damp * p.backward + s * s * bm * div(v) - \
            s * s * (w0 / qp) * h
        u_p = Eq(p.forward, damp * pde_p)

        return [u_h, u_v, u_w, u_p]

    else:

        # Auxiliary functions
        h = TimeFunction(name="h", grid=model.grid, time_order=2, space_order=space_order,
                         staggered=NODE)
        g = TimeFunction(name="g", grid=model.grid, time_order=2, space_order=space_order,
                         staggered=NODE)

        u_h = Eq(h, bm * p)

        u_v = Eq(v, b * grad(h))

        u_g = Eq(g, (p.forward - p) / s)

        pde_p = 2. * p - damp * p.forward + s * s * div(v) + \
            s * s * (w0 / qp) * g
        u_p = Eq(p.backward, damp * pde_p)

        return [u_h, u_v, u_g, u_p]


def sls(model, geometry, v, p, forward=True, **kwargs):
    """
    Stencil created from Blanch and Symes (1995) / Dutta and Schuster (2014)
    viscoacoustic wave equation.

    https://library.seg.org/doi/pdf/10.1190/1.1822695
    https://library.seg.org/doi/pdf/10.1190/geo2013-0414.1

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('sls', time_order)]
    eqn = eq_stencil(model, geometry, v, p, forward=forward, save=kwargs.get('save'))

    return eqn


def ren(model, geometry, v, p, forward=True, **kwargs):
    """
    Stencil created from Ren et al. (2014) viscoacoustic wave equation.

    https://academic.oup.com/gji/article/197/2/948/616510

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('ren', time_order)]
    eqn = eq_stencil(model, geometry, v, p, forward=forward, save=kwargs.get('save'))

    return eqn


def deng_mcmechan(model, geometry, v, p, forward=True, **kwargs):
    """
    Stencil created from Deng and McMechan (2007) viscoacoustic wave equation.

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    v : VectorTimeFunction
        Particle velocity.
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('deng_mcmechan', time_order)]
    eqn = eq_stencil(model, geometry, v, p, forward=forward, save=kwargs.get('save'))

    return eqn


def ForwardOperator(model, geometry, space_order=4, kernel='sls', time_order=2,
                    save=False, **kwargs):
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
        sls (Standard Linear Solid) :
        1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
        viscoacoustic equation
        2nd order - Bai et al. (2014) viscoacoustic equation
        ren - Ren et al. (2014) viscoacoustic equation
        deng_mcmechan - Deng and McMechan (2007) viscoacoustic equation
        Defaults to sls 2nd order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """
    dt = model.grid.time_dim.spacing
    m = model.m

    # Create symbols for forward wavefield, particle velocity, source and receivers
    # Velocity:
    v = VectorTimeFunction(name="v", grid=model.grid,
                           save=geometry.nt if save else None,
                           time_order=time_order, space_order=space_order)

    p = TimeFunction(name="p", grid=model.grid, staggered=NODE,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)

    src = PointSource(name="src", grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)

    rec = Receiver(name="rec", grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, v, p, save=save)

    # The source injection term
    src_term = src.inject(field=p.forward, expr=src * dt**2 / m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=p)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4, kernel='sls', time_order=2, **kwargs):
    """
    Construct an adjoint modelling operator in an viscoacoustic media.

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
        sls (Standard Linear Solid) :
        1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
        viscoacoustic equation
        2nd order - Bai et al. (2014) viscoacoustic equation
        ren - Ren et al. (2014) viscoacoustic equation
        deng_mcmechan - Deng and McMechan (2007) viscoacoustic equation
        Defaults to sls 2nd order.
    """
    dt = model.grid.time_dim.spacing
    m = model.m

    u = VectorTimeFunction(name="u", grid=model.grid, time_order=time_order,
                           space_order=space_order)
    q = TimeFunction(name="q", grid=model.grid, save=None, time_order=time_order,
                     space_order=space_order, staggered=NODE)
    srca = PointSource(name='srca', grid=model.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, u, q, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=q.backward, expr=rec * dt**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=q)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


kernels = {'sls': sls, 'ren': ren, 'deng_mcmechan': deng_mcmechan}
stencils = {('sls', 1): sls_1st_stencil, ('sls', 2): sls_2nd_stencil,
            ('deng_mcmechan', 1): deng_1st_stencil,
            ('deng_mcmechan', 2): deng_2nd_stencil,
            ('ren', 1): ren_1st_stencil, ('ren', 2): ren_2nd_stencil}

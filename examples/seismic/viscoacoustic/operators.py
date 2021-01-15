import sympy as sp
import numpy as np

from devito import (Eq, Operator, VectorTimeFunction, TimeFunction, NODE,
                    div, grad)
from examples.seismic import PointSource, Receiver


def src_rec(p, model, geometry, **kwargs):
    """
    Forward case: Source injection and receiver interpolation
    Adjoint case: Receiver injection and source interpolation
    """
    dt = model.grid.time_dim.spacing
    m = model.m
    # Source symbol with input wavelet
    src = PointSource(name="src", grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    forward = kwargs.get('forward', True)
    time_order = p.time_order

    if forward:
        # The source injection term
        if(time_order == 1):
            src_term = src.inject(field=p.forward, expr=src * dt)
        else:
            src_term = src.inject(field=p.forward, expr=src * dt**2 / m)
        # Create interpolation expression for receivers
        rec_term = rec.interpolate(expr=p)
    else:
        # Construct expression to inject receiver values
        if(time_order == 1):
            rec_term = rec.inject(field=p.backward, expr=rec * dt)
        else:
            rec_term = rec.inject(field=p.backward, expr=rec * dt**2 / m)
        # Create interpolation expression for the adjoint-source
        src_term = src.interpolate(expr=p)

    return src_term + rec_term


def sls_1st_order(model, geometry, p, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    from Blanch and Symes (1995) / Dutta and Schuster (2014).

    https://library.seg.org/doi/pdf/10.1190/1.1822695
    https://library.seg.org/doi/pdf/10.1190/geo2013-0414.1

    Parameters
    ----------
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

    # Particle Velocity
    v = kwargs.pop('v')

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


def sls_2nd_order(model, geometry, p, **kwargs):
    """
    Implementation of the 2nd order viscoacoustic wave-equation from Bai (2014).

    https://library.seg.org/doi/10.1190/geo2013-0030.1

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)
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

        pde_r = r + s * (tt / t_s) * rho * div(b * grad(p, shift=.5), shift=-.5) - \
            s * (1. / t_s) * r
        u_r = Eq(r.forward, damp * pde_r)

        pde_p = 2. * p - damp * p.backward + s * s * vp * vp * (1. + tt) * rho * \
            div(b * grad(p, shift=.5), shift=-.5) - s * s * vp * vp * r.forward
        u_p = Eq(p.forward, damp * pde_p)

        return [u_r, u_p]

    else:

        pde_r = r + s * (tt / t_s) * p - s * (1. / t_s) * r
        u_r = Eq(r.backward, damp * pde_r)

        pde_p = 2. * p - damp * p.forward + s * s * vp * vp * \
            div(b * grad((1. + tt) * rho * p, shift=.5), shift=-.5) - s * s * vp * vp * \
            div(b * grad(rho * r.backward, shift=.5), shift=-.5)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_r, u_p]


def ren_1st_order(model, geometry, p, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation from Ren et al. (2014).

    https://academic.oup.com/gji/article/197/2/948/616510

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Particle velocity
    v = kwargs.pop('v')

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    # Define PDE
    pde_v = v - s * b * grad(p)
    u_v = Eq(v.forward, damp * pde_v)

    pde_p = p - s * vp * vp * rho * div(v.forward) + \
        s * ((vp * vp * rho) / (w0 * qp)) * div(b * grad(p, shift=.5), shift=-.5)
    u_p = Eq(p.forward, damp * pde_p)

    return [u_v, u_p]


def ren_2nd_order(model, geometry, p, **kwargs):
    """
    Implementation of the 2nd order viscoacoustic wave-equation from Ren et al. (2014).

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)

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

        pde_p = 2. * p - damp * p.backward + s * s * bm * \
            div(b * grad(p, shift=.5), shift=-.5) + s * s * eta * rho * \
            div(b * grad(p - p.backward, shift=.5) / s, shift=-.5)

        u_p = Eq(p.forward, damp * pde_p)

        return [u_p]

    else:

        pde_p = 2. * p - damp * p.forward + s * s * \
            div(b * grad(bm * p, shift=.5), shift=-.5) - s * s * \
            div(b * grad(((p.forward - p) / s) * rho * eta, shift=.5), shift=-.5)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_p]


def deng_1st_order(model, geometry, p, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    from Deng and McMechan (2007).

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Particle velocity
    v = kwargs.pop('v')

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


def deng_2nd_order(model, geometry, p, **kwargs):
    """
    Implementation of the 2nd order viscoacoustic wave-equation
    from Deng and McMechan (2007).

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)

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

        pde_p = 2. * p - damp*p.backward + s * s * bm * \
            div(b * grad(p, shift=.5), shift=-.5) - s * s * w0/qp * (p - p.backward)/s
        u_p = Eq(p.forward, damp * pde_p)

        return [u_p]

    else:

        pde_p = 2. * p - damp * p.forward + s * s * w0 / qp * (p.forward - p) / s + \
            s * s * div(b * grad(bm * p, shift=.5), shift=-.5)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_p]


def sls(model, geometry, p, forward=True, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    from Blanch and Symes (1995) / Dutta and Schuster (2014) and
    Implementation of the 2nd order viscoacoustic wave-equation from Bai (2014).

    https://library.seg.org/doi/pdf/10.1190/1.1822695
    https://library.seg.org/doi/pdf/10.1190/geo2013-0414.1
    https://library.seg.org/doi/10.1190/geo2013-0030.1

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('sls', time_order)]
    eqn = eq_stencil(model, geometry, p, forward=forward, **kwargs)

    return eqn


def ren(model, geometry, p, forward=True, **kwargs):
    """
    Implementation of the 1st and 2nd order viscoacoustic wave-equation from
    Ren et al. (2014).

    https://academic.oup.com/gji/article/197/2/948/616510
    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('ren', time_order)]
    eqn = eq_stencil(model, geometry, p, forward=forward, **kwargs)

    return eqn


def deng_mcmechan(model, geometry, p, forward=True, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation and 2nd order
    viscoacoustic wave-equation from Deng and McMechan (2007).

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('deng_mcmechan', time_order)]
    eqn = eq_stencil(model, geometry, p, forward=forward, **kwargs)

    return eqn


def ForwardOperator(model, geometry, space_order=4, kernel='sls', time_order=2,
                    save=False, **kwargs):
    """
    Construct method for the forward modelling operator in a viscoacoustic medium.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : string, optional
        selects a viscoacoustic equation from the options below:
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
    # Create symbols for forward wavefield, particle velocity, source and receivers

    if time_order == 1:
        v = VectorTimeFunction(name="v", grid=model.grid,
                               save=geometry.nt if save else None,
                               time_order=time_order, space_order=space_order)
        kwargs.update({'v': v})

    p = TimeFunction(name="p", grid=model.grid, staggered=NODE,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, p, save=save, **kwargs)

    srcrec = src_rec(p, model, geometry)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + srcrec, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4, kernel='sls', time_order=2, **kwargs):
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
    kernel : selects a visco-acoustic equation from the options below:
        sls (Standard Linear Solid) :
        1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
        viscoacoustic equation
        2nd order - Bai et al. (2014) viscoacoustic equation
        ren - Ren et al. (2014) viscoacoustic equation
        deng_mcmechan - Deng and McMechan (2007) viscoacoustic equation
        Defaults to sls 2nd order.
    """
    pa = TimeFunction(name="pa", grid=model.grid, save=None, time_order=time_order,
                      space_order=space_order, staggered=NODE)

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, pa, forward=False)

    srcrec = src_rec(pa, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + srcrec, subs=model.spacing_map, name='Adjoint', **kwargs)


kernels = {'sls': sls, 'ren': ren, 'deng_mcmechan': deng_mcmechan}
stencils = {('sls', 1): sls_1st_order, ('sls', 2): sls_2nd_order,
            ('deng_mcmechan', 1): deng_1st_order,
            ('deng_mcmechan', 2): deng_2nd_order,
            ('ren', 1): ren_1st_order, ('ren', 2): ren_2nd_order}

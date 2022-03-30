import sympy as sp
import numpy as np

from devito import (Eq, Operator, VectorTimeFunction, TimeFunction, Function, NODE,
                    div, grad, solve)
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

    scale = dt / m if time_order == 1 else dt**2 / m

    if forward:
        # The source injection term
        src_term = src.inject(field=p.forward, expr=src * scale)
        # Create interpolation expression for receivers
        rec_term = rec.interpolate(expr=p)
    else:
        # Construct expression to inject receiver values
        rec_term = rec.inject(field=p.backward, expr=rec * scale)
        # Create interpolation expression for the adjoint-source
        src_term = src.interpolate(expr=p)

    return src_term + rec_term


def sls_1st_order(model, geometry, p, r=None, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation based on SLS model from:
    1 - Jose M. Carcione (2015): Wave Fields in Real Media: Wave Propagation
    in Anisotropic, Anelastic, Porous and Electromagnetic Media
    2 - Blanch and Symes (1995)
    3 - Dutta and Schuster (2014)

    https://www.elsevier.com/books/wave-fields-in-real-media/carcione/978-0-08-099999-9
    https://library.seg.org/doi/pdf/10.1190/1.1822695
    https://library.seg.org/doi/pdf/10.1190/geo2013-0414.1

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    r : TimeFunction
        Memory variable.
    """
    forward = kwargs.get('forward', True)
    space_order = p.space_order
    save = kwargs.get('save', False)
    save_t = geometry.nt if save else None
    b = model.b
    m = model.m
    damp = model.damp
    qp = model.qp
    f0 = geometry._f0
    q = kwargs.get('q', 0)

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

    # Attenuation Memory variable
    r = r or TimeFunction(name="r", grid=model.grid, time_order=1,
                          space_order=space_order, save=save_t, staggered=NODE)

    if forward:

        # Particle velocity
        pde_v = v.dt + b * grad(p)
        u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

        # Attenuation Memory variable
        pde_r = r.dt + (1. / t_s) * (r + tt * rho * div(v.forward))
        u_r = Eq(r.forward, damp * solve(pde_r, r.forward))

        # Pressure
        pde_p = m * p.dt + rho * (tt + 1.) * div(v.forward) + r.forward - q
        u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

        return [u_v, u_r, u_p]

    else:

        # Attenuation Memory variable
        pde_r = r.dt.T + (1. / t_s) * r + p
        u_r = Eq(r.backward, damp * solve(pde_r, r.backward))

        # Particle velocity
        # Because v is a Vector, `.T` applies a standard matrix transpose
        # so we need to do the derivative transpose by hand with `-*.dtl`
        pde_v = -v.dtl - grad(rho * (1. + tt) * p) - \
            grad((1. / t_s) * rho * tt * r.backward)
        u_v = Eq(v.backward, damp * solve(pde_v, v.backward))

        # Pressure
        pde_p = m * p.dt.T - div(b * v.backward)
        u_p = Eq(p.backward, damp * solve(pde_p, p.backward))

        return [u_r, u_v, u_p]


def sls_2nd_order(model, geometry, p, r=None, **kwargs):
    """
    Implementation of the 2nd order viscoacoustic wave-equation based on SLS model from:
    1 - Jose M. Carcione (2015): Wave Fields in Real Media: Wave Propagation
    in Anisotropic, Anelastic, Porous and Electromagnetic Media
    2 - Bai (2014).

    https://www.elsevier.com/books/wave-fields-in-real-media/carcione/978-0-08-099999-9
    https://library.seg.org/doi/10.1190/geo2013-0030.1

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    r : TimeFunction
        Attenuation Memory variable.
    """
    forward = kwargs.get('forward', True)
    space_order = p.space_order
    save = kwargs.get('save', False)
    save_t = geometry.nt if save else None
    b = model.b
    m = model.m
    damp = model.damp
    qp = model.qp
    f0 = geometry._f0
    q = kwargs.get('q', 0)

    # The stress relaxation parameter
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0

    # The strain relaxation parameter
    t_ep = 1./(f0**2*t_s)

    # The relaxation time
    tt = (t_ep/t_s)-1.

    # Density
    rho = 1. / b

    # Attenuation Memory variable
    r = r or TimeFunction(name="r", grid=model.grid, time_order=2,
                          space_order=space_order, save=save_t, staggered=NODE)

    if forward:
        # Attenuation Memory variable
        pde_r = r.dt - (tt / t_s) * rho * div(b * grad(p, shift=.5), shift=-.5) + \
            (1. / t_s) * r
        u_r = Eq(r.forward, damp * solve(pde_r, r.forward))

        # Pressure
        pde_p = m * p.dt2 - rho * (1. + tt) * div(b * grad(p, shift=.5), shift=-.5) + \
            r.forward - q + (1 - damp) * p.dt
        u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

        return [u_r, u_p]

    else:
        # Attenuation Memory variable
        pde_r = r.dt.T + (tt / t_s) * p + (1. / t_s) * r
        u_r = Eq(r.backward, damp * solve(pde_r, r.backward))

        # Pressure
        pde_p = m * p.dt2 - div(b * grad((1. + tt) * rho * p, shift=.5), shift=-.5) - \
            div(b * grad(rho * r.backward, shift=.5), shift=-.5) + \
            (1 - damp) * p.dt.T
        u_p = Eq(p.backward, damp * solve(pde_p, p.backward))

        return [u_r, u_p]


def kv_1st_order(model, geometry, p, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    based on Kelvin-Voigt model from:
    1 - Jose M. Carcione (2015): Wave Fields in Real Media: Wave Propagation
    in Anisotropic, Anelastic, Porous and Electromagnetic Media

    https://www.elsevier.com/books/wave-fields-in-real-media/carcione/978-0-08-099999-9


    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)
    f0 = geometry._f0
    m = model.m
    b = model.b
    qp = model.qp
    damp = model.damp

    # Particle velocity
    v = kwargs.pop('v')

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b
    tau = 1 / (w0 * qp)

    if forward:
        # Particle velocity
        pde_v = v.dt + b * grad(p)
        u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

        # Pressure
        pde_p = m * p.dt + rho * div(v.forward) - \
            tau * rho * div(b * grad(p, shift=.5), shift=-.5)
        u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

        return [u_v, u_p]
    else:
        # Particle velocity
        # Becaue v is a Vector, `.T` applies a standard matrix transpose
        # so we need to do the derivative transpose by hand with `-*.dtl`
        pde_v = -v.dtl - grad(rho * p)
        u_v = Eq(v.backward, damp * solve(pde_v, v.backward))

        # Pressure
        pde_p = m * p.dt.T - div(b * grad(rho * tau * p, shift=.5), shift=-.5) - \
            div(b * v.backward)
        u_p = Eq(p.backward, damp * solve(pde_p, p.backward))

        return [u_v, u_p]


def kv_2nd_order(model, geometry, p, **kwargs):
    """
    Implementation of the 2nd order viscoacoustic wave-equation
    based on Kelvin-Voigt model from:
    1 - Jose M. Carcione (2015): Wave Fields in Real Media: Wave Propagation
    in Anisotropic, Anelastic, Porous and Electromagnetic Media
    2 - Ren et al. (2014).

    https://www.elsevier.com/books/wave-fields-in-real-media/carcione/978-0-08-099999-9
    https://library.seg.org/doi/pdf/10.1190/1.2714334


    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)

    s = model.grid.stepping_dim.spacing
    t0 = p.indices[0] - s / 2
    f0 = geometry._f0
    m = model.m
    b = model.b
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b
    tau = 1 / (w0 * qp)

    if forward:
        # Pressure
        pde_p = m * p.dt2 - rho * div(b * grad(p, shift=.5), shift=-.5) - \
            tau * rho * div(b * grad(p.dt(x0=t0), shift=.5), shift=-.5) + \
            (1 - damp) * p.dt

        u_p = Eq(p.forward, solve(pde_p, p.forward))

        return [u_p]

    else:
        # Pressure
        pde_p = m * p.dt2 - div(b * grad(rho * p, shift=.5), shift=-.5) - \
            div(b * grad(rho * tau * p.dt(x0=t0).T, shift=.5), shift=-.5) + \
            (1 - damp) * p.dt.T
        u_p = Eq(p.backward, solve(pde_p, p.backward))

        return [u_p]


def maxwell_1st_order(model, geometry, p, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    based on Maxwell model from:
    1 - Jose M. Carcione (2015): Wave Fields in Real Media: Wave Propagation
    in Anisotropic, Anelastic, Porous and Electromagnetic Media

    https://www.elsevier.com/books/wave-fields-in-real-media/carcione/978-0-08-099999-

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)
    f0 = geometry._f0
    m = model.m
    b = model.b
    qp = model.qp
    damp = model.damp

    # Particle velocity
    v = kwargs.pop('v')

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    if forward:
        # Particle velocity
        pde_v = v.dt + b * grad(p)
        u_v = Eq(v.forward, damp * solve(pde_v, v.forward))
        # Pressure
        pde_p = m * p.dt + rho * div(v.forward) + (w0 / qp) * p
        u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

        return [u_v, u_p]

    else:
        # Particle velocity
        # Becaue v is a Vector, `.T` applies a standard matrix transpose
        # so we need to do the derivative transpose by hand with `-*.dtl`
        pde_v = -v.dtl - grad(rho * p)
        u_v = Eq(v.backward, damp * solve(pde_v, v.backward))
        # Pressure
        pde_p = m * p.dt.T - div(b * v.backward) + (w0 / qp) * p
        u_p = Eq(p.backward, damp * solve(pde_p, p.backward))

        return [u_v, u_p]


def maxwell_2nd_order(model, geometry, p, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    based on Maxwell model from:
    1 - Jose M. Carcione (2015): Wave Fields in Real Media: Wave Propagation
    in Anisotropic, Anelastic, Porous and Electromagnetic Media
    2 - Deng and McMechan (2007).

    https://www.elsevier.com/books/wave-fields-in-real-media/carcione/978-0-08-099999-9
    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)

    s = model.grid.stepping_dim.spacing
    t0 = p.indices[0] - s / 2
    f0 = geometry._f0
    b = model.b
    m = model.m
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    if forward:
        # Pressure
        pde_p = m * p.dt2 - rho * div(b * grad(p, shift=.5), shift=-.5) + \
            m * w0 / qp * p.dt(x0=t0) + (1 - damp) * p.dt
        u_p = Eq(p.forward, solve(pde_p, p.forward))

        return [u_p]

    else:
        # Pressure
        pde_p = m * p.dt2 + m * w0 / qp * p.dt(x0=t0).T + (1 - damp) * p.dt.T - \
            div(b * grad(rho * p, shift=.5), shift=-.5)
        u_p = Eq(p.backward, solve(pde_p, p.backward))

        return [u_p]


def sls(model, geometry, p, forward=True, **kwargs):
    """
    Implementation of the viscoacoustic wave-equation from:
    1 - Jose M. Carcione (2015): Wave Fields in Real Media: Wave Propagation
    in Anisotropic, Anelastic, Porous and Electromagnetic Media
    2 - Blanch and Symes (1995) / Dutta and Schuster (2014) and
    3 - Bai (2014).

    https://www.elsevier.com/books/wave-fields-in-real-media/carcione/978-0-08-099999-9
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


def kv(model, geometry, p, forward=True, **kwargs):
    """
    Implementation of the 1st and 2nd order viscoacoustic wave-equation from:
    1 - Jose M. Carcione (2015): Wave Fields in Real Media: Wave Propagation
    in Anisotropic, Anelastic, Porous and Electromagnetic Media
    2 - Ren et al. (2014).

    https://www.elsevier.com/books/wave-fields-in-real-media/carcione/978-0-08-099999-9
    https://academic.oup.com/gji/article/197/2/948/616510
    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('kv', time_order)]
    eqn = eq_stencil(model, geometry, p, forward=forward, **kwargs)

    return eqn


def maxwell(model, geometry, p, forward=True, **kwargs):
    """
    Implementation of the 1st and 2nd order viscoacoustic wave-equation from:
    1 - Jose M. Carcione (2015): Wave Fields in Real Media: Wave Propagation
    in Anisotropic, Anelastic, Porous and Electromagnetic Media
    2 - Deng and McMechan (2007).

    https://www.elsevier.com/books/wave-fields-in-real-media/carcione/978-0-08-099999-9
    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('maxwell', time_order)]
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
        SLS (Standard Linear Solid) :
        1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
        viscoacoustic equation
        2nd order - Bai et al. (2014) viscoacoustic equation
        kv - Ren et al. (2014) viscoacoustic equation
        maxwell - Deng and McMechan (2007) viscoacoustic equation
        Defaults to sls 2nd order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """
    # Create symbols for forward wavefield, particle velocity, source and receivers
    save_t = geometry.nt if save else None

    if time_order == 1:
        v = VectorTimeFunction(name="v", grid=model.grid, time_order=time_order,
                               space_order=space_order, save=save_t)
        kwargs.update({'v': v})

    p = TimeFunction(name="p", grid=model.grid, time_order=time_order,
                     space_order=space_order, save=save_t, staggered=NODE)

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, p, save=save, **kwargs)

    src_term, rec_term = src_rec(p, model, geometry)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4, kernel='SLS', time_order=2, **kwargs):
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
        kv - Ren et al. (2014) viscoacoustic equation
        maxwell - Deng and McMechan (2007) viscoacoustic equation
        Defaults to sls 2nd order.
    """
    if time_order == 1:
        va = VectorTimeFunction(name="va", grid=model.grid, time_order=time_order,
                                space_order=space_order)
        kwargs.update({'v': va})

    pa = TimeFunction(name="pa", grid=model.grid, time_order=time_order,
                      space_order=space_order, staggered=NODE)

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, pa, forward=False, **kwargs)

    src_term, rec_term = src_rec(pa, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def GradientOperator(model, geometry, space_order=4, kernel='sls', time_order=2,
                     save=True, **kwargs):
    """
    Construct a gradient operator in an acoustic media.

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
    kernel : selects a visco-acoustic equation from the options below:
        sls (Standard Linear Solid) :
        1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
        viscoacoustic equation
        2nd order - Bai et al. (2014) viscoacoustic equation
        kv - Ren et al. (2014) viscoacoustic equation
        maxwell - Deng and McMechan (2007) viscoacoustic equation
        Defaults to sls 2nd order.
    """
    # Gradient symbol and wavefield symbols
    save_t = geometry.nt if save else None

    grad = Function(name='grad', grid=model.grid)
    p = TimeFunction(name='p', grid=model.grid, time_order=time_order,
                     space_order=space_order, save=save_t, staggered=NODE)
    pa = TimeFunction(name='pa', grid=model.grid, time_order=time_order,
                      space_order=space_order, staggered=NODE)

    if time_order == 1:
        va = VectorTimeFunction(name="va", grid=model.grid, time_order=time_order,
                                space_order=space_order)
        kwargs.update({'v': va})

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, pa, forward=False, save=False, **kwargs)

    if time_order == 1:
        gradient_update = Eq(grad, grad - p.dt * pa)
    else:
        gradient_update = Eq(grad, grad + p.dt * pa.dt)

    # Add expression for receiver injection
    _, recterm = src_rec(pa, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + recterm + [gradient_update], subs=model.spacing_map,
                    name='Gradient', **kwargs)


def BornOperator(model, geometry, space_order=4, kernel='sls', time_order=2, **kwargs):
    """
    Construct an Linearized Born operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    # Create wavefields and a dm field
    p = TimeFunction(name='p', grid=model.grid, time_order=time_order,
                     space_order=space_order, staggered=NODE)
    P = TimeFunction(name='P', grid=model.grid, time_order=time_order,
                     space_order=space_order, staggered=NODE)
    rp = TimeFunction(name="rp", grid=model.grid, time_order=time_order,
                      space_order=space_order, staggered=NODE)
    rP = TimeFunction(name="rP", grid=model.grid, time_order=time_order,
                      space_order=space_order, staggered=NODE)
    dm = Function(name='dm', grid=model.grid, space_order=0)

    if time_order == 1:
        v = VectorTimeFunction(name="v", grid=model.grid,
                               time_order=time_order, space_order=space_order)
        kwargs.update({'v': v})

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn1 = eq_kernel(model, geometry, p, r=rp, **kwargs)

    if time_order == 1:
        dv = VectorTimeFunction(name="dv", grid=model.grid,
                                time_order=time_order, space_order=space_order)
        kwargs.update({'v': dv})

        q = -dm * p.dt
    else:
        q = -dm * p.dt2

    eqn2 = eq_kernel(model, geometry, P, r=rP, q=q, **kwargs)

    # Add source term expression for p
    src_term, _ = src_rec(p, model, geometry)

    # Create receiver interpolation expression from P
    _, rec_term = src_rec(P, model, geometry)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + src_term + rec_term + eqn2, subs=model.spacing_map,
                    name='Born', **kwargs)


kernels = {'sls': sls, 'kv': kv, 'maxwell': maxwell}
stencils = {('sls', 1): sls_1st_order, ('sls', 2): sls_2nd_order,
            ('maxwell', 1): maxwell_1st_order,
            ('maxwell', 2): maxwell_2nd_order,
            ('kv', 1): kv_1st_order, ('kv', 2): kv_2nd_order}

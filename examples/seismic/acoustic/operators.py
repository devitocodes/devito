from sympy import Eq
from sympy.abc import h, s

from devito import Operator, Forward, Backward, DenseData, TimeData, time
from examples.seismic import PointSource, Receiver


def ForwardOperator(model, source, receiver, time_order=2, space_order=4,
                    save=False, **kwargs):
    """
    Constructor method for the forward modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param time_order: Time discretization order
    :param space_order: Space discretization order
    :param save : Saving flag, True saves all time steps, False only the three
    """
    m, damp = model.m, model.damp

    # Create symbols for forward wavefield, source and receivers
    u = TimeData(name='u', shape=model.shape_domain, time_dim=source.nt,
                 time_order=time_order, space_order=space_order, save=save,
                 dtype=model.dtype)
    src = PointSource(name='src', ntime=source.nt, ndim=source.ndim,
                      npoint=source.npoint)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    if time_order == 2:
        biharmonic = 0
        dt = model.critical_dt
    else:
        biharmonic = u.laplace2(1/m)
        dt = 1.73 * model.critical_dt

    # Derive both stencils from symbolic equation:
    # Create the stencil by hand instead of calling numpy solve for speed purposes
    # Simple linear solve of a u(t+dt) + b u(t) + c u(t-dt) = L for u(t+dt)
    stencil = 1 / (2 * m + s * damp) * (
        4 * m * u + (s * damp - 2 * m) * u.backward +
        2 * s**2 * (u.laplace + s**2 / 12 * biharmonic))
    eqn = [Eq(u.forward, stencil)]

    # Construct expression to inject source values
    # Note that src and field terms have differing time indices:
    #   src[time, ...] - always accesses the "unrolled" time index
    #   u[ti + 1, ...] - accesses the forward stencil value
    ti = u.indices[0]
    src_term = src.inject(field=u, u_t=ti + 1, offset=model.nbpml,
                          expr=src * dt**2 / m, p_t=time)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u, u_t=ti, offset=model.nbpml)

    return Operator(eqn + src_term + rec_term,
                    subs={s: dt, h: model.get_spacing()},
                    time_axis=Forward, name='Forward', **kwargs)


def AdjointOperator(model, source, receiver, time_order=2, space_order=4, **kwargs):
    """
    Constructor method for the adjoint modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param time_order: Time discretization order
    :param space_order: Space discretization order
    """
    m, damp = model.m, model.damp

    v = TimeData(name='v', shape=model.shape_domain, save=False,
                 time_order=time_order, space_order=space_order,
                 dtype=model.dtype)
    srca = PointSource(name='srca', ntime=source.nt, ndim=source.ndim,
                       npoint=source.npoint)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    if time_order == 2:
        biharmonic = 0
        dt = model.critical_dt
    else:
        biharmonic = v.laplace2(1/m)
        dt = 1.73 * model.critical_dt

    # Derive both stencils from symbolic equation
    stencil = 1 / (2 * m + s * damp) * (
        4 * m * v + (s * damp - 2 * m) * v.forward +
        2 * s**2 * (v.laplace + s**2 / 12 * biharmonic))
    eqn = Eq(v.backward, stencil)

    # Construct expression to inject receiver values
    ti = v.indices[0]
    receivers = rec.inject(field=v, u_t=ti - 1, offset=model.nbpml,
                           expr=rec * dt**2 / m, p_t=time)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v, u_t=ti, offset=model.nbpml)

    return Operator([eqn] + receivers + source_a,
                    subs={s: dt, h: model.get_spacing()},
                    time_axis=Backward, name='Adjoint', **kwargs)


def GradientOperator(model, source, receiver, time_order=2, space_order=4, **kwargs):
    """
    Constructor method for the gradient operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param time_order: Time discretization order
    :param space_order: Space discretization order
    """
    m, damp = model.m, model.damp

    # Gradient symbol and wavefield symbols
    grad = DenseData(name='grad', shape=model.shape_domain,
                     dtype=model.dtype)
    u = TimeData(name='u', shape=model.shape_domain, save=True,
                 time_dim=source.nt, time_order=time_order,
                 space_order=space_order, dtype=model.dtype)
    v = TimeData(name='v', shape=model.shape_domain, save=False,
                 time_order=time_order, space_order=space_order,
                 dtype=model.dtype)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    if time_order == 2:
        biharmonic = 0
        dt = model.critical_dt
        gradient_update = Eq(grad, grad - u.dt2 * v)
    else:
        biharmonic = v.laplace2(1/m)
        biharmonicu = - u.laplace2(1/(m**2))
        dt = 1.73 * model.critical_dt
        gradient_update = Eq(grad, grad - (u.dt2 - s**2 / 12.0 * biharmonicu) * v)

    # Derive stencil from symbolic equation
    stencil = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * v + (s * damp - 2.0 * m) *
         v.forward + 2.0 * s ** 2 * (v.laplace + s**2 / 12.0 * biharmonic))
    eqn = Eq(v.backward, stencil)

    # Add expression for receiver injection
    ti = v.indices[0]
    receivers = rec.inject(field=v, u_t=ti - 1, offset=model.nbpml,
                           expr=rec * dt * dt / m, p_t=time)

    return Operator([eqn] + [gradient_update] + receivers,
                    subs={s: dt, h: model.get_spacing()},
                    time_axis=Backward, name='Gradient', **kwargs)


def BornOperator(model, source, receiver, time_order=2, space_order=4, **kwargs):
    """
    Constructor method for the Linearized Born operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param time_order: Time discretization order
    :param space_order: Space discretization order
    """
    m, damp = model.m, model.damp

    # Create source and receiver symbols
    src = PointSource(name='src', ntime=source.nt, ndim=source.ndim,
                      npoint=source.npoint)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    # Create wavefields and a dm field
    u = TimeData(name="u", shape=model.shape_domain, save=False,
                 time_order=time_order, space_order=space_order,
                 dtype=model.dtype)
    U = TimeData(name="U", shape=model.shape_domain, save=False,
                 time_order=time_order, space_order=space_order,
                 dtype=model.dtype)
    dm = DenseData(name="dm", shape=model.shape_domain,
                   dtype=model.dtype)

    if time_order == 2:
        biharmonicu = 0
        biharmonicU = 0
        dt = model.critical_dt
    else:
        biharmonicu = u.laplace2(1/m)
        biharmonicU = U.laplace2(1/m)
        dt = 1.73 * model.critical_dt

    # Derive both stencils from symbolic equation
    # first_eqn = m * u.dt2 - u.laplace + damp * u.dt
    # second_eqn = m * U.dt2 - U.laplace - dm* u.dt2 + damp * U.dt
    stencil1 = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * u + (s * damp - 2.0 * m) *
         u.backward + 2.0 * s ** 2 * (u.laplace + s**2 / 12 * biharmonicu))
    stencil2 = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * U + (s * damp - 2.0 * m) *
         U.backward + 2.0 * s ** 2 * (U.laplace +
                                      s**2 / 12 * biharmonicU - dm * u.dt2))
    eqn1 = Eq(u.forward, stencil1)
    eqn2 = Eq(U.forward, stencil2)

    # Add source term expression for u
    ti = u.indices[0]
    source = src.inject(field=u, u_t=ti + 1, offset=model.nbpml,
                        expr=src * dt * dt / m, p_t=time)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U, u_t=ti, offset=model.nbpml)

    return Operator([eqn1] + source + [eqn2] + receivers,
                    subs={s: dt, h: model.get_spacing()},
                    time_axis=Forward, name='Born', **kwargs)

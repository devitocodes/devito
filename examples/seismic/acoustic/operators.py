from sympy import Eq, diff, solve, Symbol

from devito import Operator, Forward, Backward, DenseData, TimeData, time, t
from devito.logger import error
from examples.seismic import PointSource, Receiver, ABC


def laplacian(field, time_order, m, s):
    """
    Spacial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))
    :param field:  Symbolic TimeData object, solution to be computed
    :param time_order: time order
    :param m: square slowness
    :param s: symbol of for the time-step
    :return: H
    """
    if time_order == 2:
          biharmonic = 0
    elif time_order == 4:
      biharmonic = field.laplace2(1 / m)
    else:
      error("Unsupported time order %d, order has to be 2 or 4" %
                           time_order)
    return field.laplace + s ** 2 / 12 * biharmonic


def iso_stencil(field, time_order, m, s, damp, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0
    :param field: Symbolic TimeData object, solution to be computed
    :param time_order: time order
    :param m: square slowness
    :param s: symbol of for the time-step
    :param damp: ABC dampening field (DenseData)
    :param kwargs: forwad/backward wave equation (sign of u.dt will change accordingly
    as well as the updated time-step (u.forwad or u.backward)
    :return: Stencil for the wave-equation
    """
    # Creat a temporary symbol for H to avoid expensive sympy solve
    H = Symbol('H')
    # Define time sep to be updated
    next = field.forward if kwargs.get('forward', True) else field.backward
    # Define PDE
    eq = m * field.dt2 - H + kwargs.get('q', 0)
    # Add dampening field according to the propagation direction
    eq += damp * field.dt if kwargs.get('forward', True) else -damp * field.dt
    # Solve the symbolic equation for the field to be updated
    eq_time = solve(eq, next, rational=False, simplify=False)[0]
    # Get the spacial FD
    lap = laplacian(field, time_order, m, s)
    # return the Stencil with H replaced by its symbolic expression

    return [Eq(next, eq_time.subs({H: lap}))]


def ForwardOperator(model, source, receiver, time_order=2, space_order=4,
                    save=False, **kwargs):
    """
    Constructor method for the forward modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param time_order: Time discretization order
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    m = model.m

    # Create symbols for forward wavefield, source and receivers
    u = TimeData(name='u', shape=model.shape_domain, time_dim=source.nt,
                 time_order=time_order, space_order=space_order, save=save,
                 dtype=model.dtype)
    src = PointSource(name='src', ntime=source.nt, ndim=source.ndim,
                      npoint=source.npoint)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    s = t.spacing

    # Get computational time-step value
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
    eqn = iso_stencil(u, time_order, m, s, model.damp)
    # Construct expression to inject source values
    # Note that src and field terms have differing time indices:
    #   src[time, ...] - always accesses the "unrolled" time index
    #   u[ti + 1, ...] - accesses the forward stencil value
    ti = u.indices[0]
    src_term = src.inject(field=u.forward, offset=model.nbpml,
                          expr=src * dt**2 / m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u, offset=model.nbpml)

    BC = ABC(model, u, m)
    eq_abc = BC.abc
    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.get_spacing()[j]) for i, j
                 in zip(u.indices[1:], range(len(model.shape)))])

    return Operator(eqn + src_term + rec_term + eq_abc,
                    subs=subs,
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
    m = model.m

    v = TimeData(name='v', shape=model.shape_domain, save=False,
                 time_order=time_order, space_order=space_order,
                 dtype=model.dtype)
    srca = PointSource(name='srca', ntime=source.nt, ndim=source.ndim,
                       npoint=source.npoint)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    s = t.spacing

    # Get computational time-step value
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
    eqn = iso_stencil(v, time_order, m, s, model.damp, forward=False)

    # Construct expression to inject receiver values
    ti = v.indices[0]
    receivers = rec.inject(field=v.backward, offset=model.nbpml,
                           expr=rec * dt**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v, offset=model.nbpml)

    BC = ABC(model, v, m, taxis=Backward)
    eq_abc = BC.abc

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.get_spacing()[j]) for i, j
                 in zip(v.indices[1:], range(len(model.shape)))])

    return Operator(eqn + eq_abc + receivers + source_a,
                    subs=subs,
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
    m = model.m

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

    s = t.spacing

    # Get computational time-step value
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
    eqn = iso_stencil(v, time_order, m, s, model.damp, forward=False)
    gradient_update = Eq(grad, grad - u.dt2 * v)

    # Add expression for receiver injection
    ti = v.indices[0]
    receivers = rec.inject(field=v.backward, offset=model.nbpml,
                           expr=rec * dt * dt / m)

    BC = ABC(model, v, m, taxis=Backward)
    eq_abc = BC.abc

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.get_spacing()[j]) for i, j
                 in zip(v.indices[1:], range(len(model.shape)))])

    return Operator(eqn + receivers + eq_abc + [gradient_update],
                    subs=subs, dse='aggressive',
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
    m = model.m

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

    s = t.spacing

    # Get computational time-step value
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
    eqnu = iso_stencil(u, time_order, m, s, model.damp)
    eqnU = iso_stencil(u, time_order, m, s, model.damp, q=-dm*u.dt2)

    # Add source term expression for u
    ti = u.indices[0]
    source = src.inject(field=u.forward, offset=model.nbpml,
                        expr=src * dt * dt / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U, offset=model.nbpml)

    BC = ABC(model, u, m)
    eq_abcu = BC.abc

    BC = ABC(model, U, m)
    eq_abcU = BC.abc

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.get_spacing()[j]) for i, j
                 in zip(u.indices[1:], range(len(model.shape)))])
    return Operator(eqnu + eq_abcu + source + eqnU + eq_abcU + receivers,
                    subs=subs,
                    time_axis=Forward, name='Born', **kwargs)

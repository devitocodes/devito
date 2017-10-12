from sympy import Eq, solve, Symbol

from devito import Operator, Forward, Backward, Function, TimeFunction, time, t
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
        error("Unsupported time order %d, order has to be 2 or 4" % time_order)
    return field.laplace + s ** 2 / 12 * biharmonic


def iso_stencil(field, time_order, m, s, **kwargs):
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
    eq = m * field.dt2 - H - kwargs.get('q', 0)
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
    u = TimeFunction(name='u', grid=model.grid,
                     save=save, time_dim=source.nt if save else None,
                     time_order=2, space_order=space_order)
    src = PointSource(name='src', ntime=source.nt, ndim=source.ndim,
                      npoint=source.npoint)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    s = t.spacing
    # Get computational time-step value
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)

    eqn = iso_stencil(u, time_order, m, s)
    # Construct expression to inject source values
    # Note that src and field terms have differing time indices:
    #   src[time, ...] - always accesses the "unrolled" time index
    #   u[ti + 1, ...] - accesses the forward stencil value
    src_term = src.inject(field=u.forward, offset=model.nbpml,
                          expr=src * dt**2 / m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u, offset=model.nbpml)

    BC = ABC(model, u, m)
    eq_abc = BC.abc
    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.spacing[j]) for i, j
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

    v = TimeFunction(name='v', grid=model.grid, save=False,
                     time_order=2, space_order=space_order)

    srca = PointSource(name='srca', ntime=source.nt, ndim=source.ndim,
                       npoint=source.npoint)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    s = t.spacing
    # Get computational time-step value
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)

    eqn = iso_stencil(v, time_order, m, s, damp, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, offset=model.nbpml,
                           expr=rec * dt**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v, offset=model.nbpml)

    BC = ABC(model, v, m, taxis=Backward)
    eq_abc = BC.abc

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.spacing[j]) for i, j
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
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=True, time_dim=source.nt,
                     time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=False,
                     time_order=2, space_order=space_order)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    s = t.spacing
    # Get computational time-step value
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)

    eqn = iso_stencil(v, time_order, m, s, damp, forward=False)

    if time_order == 2:
        gradient_update = Eq(grad, grad - u.dt2 * v)
    else:
        gradient_update = Eq(grad, grad - (u.dt2 +
                                           s**2 / 12.0 * u.laplace2(m**(-2))) * v)

    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, offset=model.nbpml,
                           expr=rec * dt * dt / m)

    BC = ABC(model, v, m, taxis=Backward)
    eq_abc = BC.abc

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.spacing[j]) for i, j
                 in zip(v.indices[1:], range(len(model.shape)))])

    return Operator(eqn + receivers + eq_abc + [gradient_update],
                    subs=subs, time_axis=Backward, name='Gradient', **kwargs)


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
    u = TimeFunction(name="u", grid=model.grid, save=False,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=False,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid)

    s = t.spacing
    # Get computational time-step value
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)

    eqn1 = iso_stencil(u, time_order, m, s)
    eqn2 = iso_stencil(U, time_order, m, s, damp, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, offset=model.nbpml,
                        expr=src * dt * dt / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U, offset=model.nbpml)

    BC = ABC(model, u, m)
    eq_abcu = BC.abc

    BCU = ABC(model, U, m)
    eq_abcU = BCU.abc

    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.spacing[j]) for i, j
                 in zip(u.indices[1:], range(len(model.shape)))])
    return Operator(eqn1 + eq_abcu + source + eqn2 + eq_abcU + receivers,
                    subs=subs, time_axis=Forward, name='Born', **kwargs)

from sympy import solve, Symbol

from devito import Eq, Operator, Forward, Backward, Function, TimeFunction
from devito.logger import error
from examples.seismic import PointSource, Receiver


def laplacian(field, time_order, m, s):
    """
    Spacial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))
    :param field: Symbolic TimeFunction object, solution to be computed
    :param time_order: time order
    :param m: square slowness
    :param s: symbol for the time-step
    :return: H
    """
    if time_order == 2:
        biharmonic = 0
    elif time_order == 4:
        biharmonic = field.laplace2(1/m)
    else:
        error("Unsupported time order %d, order has to be 2 or 4" %
              time_order)
    return field.laplace + s**2/12 * biharmonic


def iso_stencil(field, time_order, m, s, damp, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0
    :param field: Symbolic TimeFunction object, solution to be computed
    :param time_order: time order
    :param m: square slowness
    :param s: symbol for the time-step
    :param damp: ABC dampening field (Function)
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
    m, damp = model.m, model.damp

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=source.nt if save else None,
                     time_order=2, space_order=space_order)
    src = PointSource(name='src', grid=model.grid, ntime=source.nt,
                      npoint=source.npoint)
    rec = Receiver(name='rec', grid=model.grid, ntime=receiver.nt,
                   npoint=receiver.npoint)

    # Get computational time-step value
    dt = model.grid.time_dim.spacing

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, time_order, m, s, damp)

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * dt**2 / m,
                          offset=model.nbpml)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u, offset=model.nbpml)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
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

    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    srca = PointSource(name='srca', grid=model.grid, ntime=source.nt,
                       npoint=source.npoint)
    rec = Receiver(name='rec', grid=model.grid, ntime=receiver.nt,
                   npoint=receiver.npoint)

    # Get computational time-step value
    dt = model.grid.time_dim.spacing

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, time_order, m, s, damp, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * dt**2 / m,
                           offset=model.nbpml)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v, offset=model.nbpml)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + source_a, subs=model.spacing_map,
                    time_axis=Backward, name='Adjoint', **kwargs)


def GradientOperator(model, source, receiver, time_order=2, space_order=4, save=True,
                     **kwargs):
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
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=source.nt if save
                     else None, time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    rec = Receiver(name='rec', grid=model.grid, ntime=receiver.nt,
                   npoint=receiver.npoint)

    # Get computational time-step value
    dt = model.grid.time_dim.spacing

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, time_order, m, s, damp, forward=False)

    if time_order == 2:
        gradient_update = Eq(grad, grad - u.dt2 * v)
    else:
        gradient_update = Eq(grad, grad - (u.dt2 +
                                           s**2 / 12.0 * u.laplace2(m**(-2))) * v)

    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, expr=rec * dt**2 / m,
                           offset=model.nbpml)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + [gradient_update], subs=model.spacing_map,
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
    src = PointSource(name='src', grid=model.grid, ntime=source.nt,
                      npoint=source.npoint)
    rec = Receiver(name='rec', grid=model.grid, ntime=receiver.nt,
                   npoint=receiver.npoint)

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid)

    # Get computational time-step value
    dt = model.grid.time_dim.spacing

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, time_order, m, s, damp)
    eqn2 = iso_stencil(U, time_order, m, s, damp, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * dt**2 / m,
                        offset=model.nbpml)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U, offset=model.nbpml)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
                    time_axis=Forward, name='Born', **kwargs)

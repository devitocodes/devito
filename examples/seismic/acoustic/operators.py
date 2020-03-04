from sympy import Symbol

from devito import Eq, Operator, Function, TimeFunction, Inc, solve
from examples.seismic import PointSource, Receiver


def laplacian(field, m, s, kernel):
    """
    Spacial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    m : Function or float
        Square slowness.
    s : float or Scalar
        The time dimension spacing.
    """
    if kernel not in ['OT2', 'OT4']:
        raise ValueError("Unrecognized kernel")

    biharmonic = field.biharmonic(1/m) if kernel == 'OT4' else 0
    return field.laplace + s**2/12 * biharmonic


def iso_stencil(field, m, s, damp, kernel, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0.

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    m : Function or float
        Square slowness.
    s : float or Scalar
        The time dimension spacing.
    damp : Function
        The damping field for absorbing boundary condition.
    forward : bool
        The propagation direction. Defaults to True.
    q : TimeFunction, Function or float
        Full-space/time source of the wave-equation.
    """

    # Creat a temporary symbol for H to avoid expensive sympy solve
    H = Symbol('H')
    # Define time sep to be updated
    next = field.forward if kwargs.get('forward', True) else field.backward
    # Define PDE
    eq = m * field.dt2 - H - kwargs.get('q', 0)
    # Add dampening field according to the propagation direction
    eq += damp * field.dt if kwargs.get('forward', True) else damp * field.dt.T
    #eq_time = solve(eq, next)
    eq_time = 2.0*(0.5*H*s**2/m + 0.5*s*damp*field/m + 1.0*field - 0.5*field.backward)/(s*damp/m + 1)
    #from IPython import embed; embed()
    # Get the spacial FD
    lap = laplacian(field, m, s, kernel)
    # return the Stencil with H replaced by its symbolic expression
    return [Eq(next, eq_time.subs({H: lap}))]


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic media.

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
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """
    m, damp = model.m, model.damp

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = PointSource(name='src', grid=geometry.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, m, s, damp, kernel)

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)
    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    kernel='OT2', **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

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
    m, damp = model.m, model.damp

    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    srca = PointSource(name='srca', grid=model.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, m, s, damp, kernel, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True,
                     kernel='OT2', **kwargs):
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
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m, damp = model.m, model.damp

    # Gradient symbol and wavefield symbols
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt if save
                     else None, time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, m, s, damp, kernel, forward=False)

    if kernel == 'OT2':
        gradient_update = Inc(grad, - u.dt2 * v)
    elif kernel == 'OT4':
        gradient_update = Inc(grad, - (u.dt2 + s**2 / 12.0 * u.biharmonic(m**(-2))) * v)
    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + [gradient_update], subs=model.spacing_map,
                    name='Gradient', **kwargs)


def BornOperator(model, geometry, space_order=4,
                 kernel='OT2', **kwargs):
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
    m, damp = model.m, model.damp

    # Create source and receiver symbols
    src = Receiver(name='src', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, m, s, damp, kernel)
    eqn2 = iso_stencil(U, m, s, damp, kernel, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
                    name='Born', **kwargs)

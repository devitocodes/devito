from devito import Eq, Function, Inc, Operator, TimeFunction, solve
from examples.seismic.free_surface import fold, mirror, solve_update, update_domain


def laplacian(field, model, kernel):
    """
    Spatial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    """
    if kernel not in ['OT2', 'OT4']:
        raise ValueError("Unrecognized kernel")
    s = model.grid.time_dim.spacing
    biharmonic = field.biharmonic(1/model.m) if kernel == 'OT4' else 0
    return field.laplace + s**2/12 * biharmonic


def time_terms(field, model, forward=True):
    """
    The time derivatives of the second-order acoustic wave equation,
    `m u.dt2 + damp u.dt`, and the field they update.

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    forward : bool, optional
        Whether to propagate forward (True) or backward (False) in time, the
        damping then differenced backward.

    Returns
    -------
    tuple
        The time derivatives, and `field.forward` or `field.backward`.
    """
    if forward:
        return model.m*field.dt2 + model.damp*field.dt, field.forward
    return model.m*field.dt2 + model.damp*field.dt.T, field.backward


def update_factor(field, model, forward=True):
    """
    The factor of the spatial operator `L` in the update of `field` solving
    `time_terms - L = 0`, e.g. `dt**2/(m + damp*dt)` backward in time, which
    the fold of a free surface needs (see `examples.seismic.free_surface`).

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    forward : bool, optional
        Whether the update is forward (True) or backward (False) in time.
    """
    terms, target = time_terms(field, model, forward)
    return solve_update(terms, 0, target)[1]


def iso_stencil(field, model, kernel, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0.

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    q : TimeFunction, Function or float
        Full-space/time source of the wave-equation.
    forward : bool, optional
        Whether to propagate forward (True) or backward (False) in time.
    """
    # Forward or backward
    forward = kwargs.get('forward', True)
    # Get the spacial FD
    lap = laplacian(field, model, kernel)
    # Get source
    q = kwargs.get('q', 0)
    # Define PDE and update rule
    terms, unext = time_terms(field, model, forward)
    eq_time = solve(terms - lap - q, unext)

    # Time-stepping stencil
    return [Eq(unext, eq_time, subdomain=update_domain(model, forward))]


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium.

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
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = geometry.src
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, model, kernel)

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + mirror(model, u.forward) + rec_term,
                    subs=model.spacing_map,
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
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m

    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    srca = geometry.new_src(name='srca', src_type=None)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    fs = fold(model, v.backward, update_factor(v, model, forward=False))
    return Operator(eqn + receivers + fs + source_a,
                    subs=model.spacing_map,
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
    m = model.m

    # Gradient symbol and wavefield symbols
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt if save
                     else None, time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    rec = geometry.rec

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    if kernel == 'OT2':
        gradient_update = Inc(grad, - u * v.dt2)
    elif kernel == 'OT4':
        gradient_update = Inc(grad, - u * v.dt2 - s**2 / 12.0 * u.biharmonic(m**(-2)) * v)
    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Substitute spacing terms to reduce flops
    fs = fold(model, v.backward, update_factor(v, model, forward=False))
    return Operator(eqn + receivers + fs + [gradient_update],
                    subs=model.spacing_map,
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
    m = model.m

    # Create source and receiver symbols
    src = geometry.src
    rec = geometry.rec

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, model, kernel)
    eqn2 = iso_stencil(U, model, kernel, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + source + mirror(model, u.forward) + eqn2 +
                    mirror(model, U.forward) + receivers, subs=model.spacing_map,
                    name='Born', **kwargs)

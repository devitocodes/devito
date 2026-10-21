import sympy

from devito import Eq, Function, Inc, Operator, TimeFunction, sign, solve
from devito.symbolics import INT, retrieve_derivatives, retrieve_functions


def freesurface(model, eq, coefficient=None):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation.

    On a model over a surface, the surface is the row below the mirror rows,
    the `fsdomain`: the field is zero on it and its odd mirror through it,
    `u(z) = -u(2 s - z)`, above it. In an adjoint update, given the
    `coefficient` of the spatial operator in it, it is the transpose of the
    mirror instead (see `_odd_fold`). Otherwise the surface is the first row,
    and the update of the rows below it, in the `fsdomain`, reads the odd
    mirror of the field in place of the points above it.

    Parameters
    ----------
    model : Model
        Physical model.
    eq : Eq
        Time-stepping stencil (time update) to mirror at the freesurface.
    coefficient : expr-like, optional
        On a model over a surface, the factor of the spatial operator in the
        adjoint update `eq` (see `operator_factor`), for its transpose.
    """
    if 'surface' in model.grid.subdomains:
        if coefficient is None:
            return _odd_mirror(model, eq.lhs)
        return _odd_fold(model, eq.lhs, coefficient)

    lhs, rhs = eq.args
    # Get vertical dimension and corresponding subdimension
    fsdomain = model.grid.subdomains['fsdomain']
    zfs = fsdomain.dimensions[-1]
    z = zfs.parent

    # Retrieve vertical derivatives
    dzs = {d for d in retrieve_derivatives(rhs) if z in d.dims}
    # Remove inner duplicate
    dzs = dzs - {d for D in dzs for d in retrieve_derivatives(D.expr) if z in d.dims}
    dzs = {d: d._eval_at(lhs).evaluate for d in dzs}

    # Finally get functions for evaluated derivatives
    funcs = {f for f in retrieve_functions(dzs.values())}

    mapper = {}
    # Antisymmetric mirror at negative indices
    # TODO: Make a proper "mirror_indices" tool function
    for f in funcs:
        zind = f.indices[-1]
        if (zind - z).as_coeff_Mul()[0] < 0:
            s = sign(zind.subs({z: zfs, z.spacing: 1}))
            mapper.update({f: s * f.subs({zind: INT(abs(zind))})})

    # Mapper for vertical derivatives
    dzmapper = {d: v.subs(mapper) for d, v in dzs.items()}

    fs_eq = [eq.func(lhs, rhs.subs(dzmapper), subdomain=fsdomain)]
    fs_eq.append(eq.func(lhs._subs(z, 0), 0, subdomain=fsdomain))

    return fs_eq


def _odd_mirror(model, field):
    """
    The free surface of `field`, e.g. `u.forward`, on a model over a surface:
    zero on the surface row `s`, and the odd mirror `u(z) = -u(2 s - z)` in the
    mirror rows above it.
    """
    subdomains = model.grid.subdomains
    z = model.grid.dimensions[-1]
    surface = subdomains['surface'].start
    return [Eq(field, 0, subdomain=subdomains['surface']),
            Eq(field, -field.subs(z, 2*surface - z), subdomain=subdomains['fsdomain'])]


def _odd_fold(model, field, coefficient):
    """
    The transpose of `_odd_mirror` in an adjoint update of `field`, e.g.
    `v.backward`, computed in the mirror rows too, where it holds the spatial
    operator times `coefficient`, in the inner product weighted by the Jacobian:
    the mirror rows folded onto the rows below the surface, which they mirror,
    `v(z) += -c/J (J v/c)(2 s - z)`, then zero, with the surface row.
    """
    grid = model.grid
    subdomains = grid.subdomains
    z = grid.dimensions[-1]
    surface = subdomains['surface'].start
    folded = (grid.jacobian*field/coefficient).subs(z, 2*surface - z)
    return [Inc(field, -coefficient/grid.jacobian*folded,
                subdomain=subdomains['mirrored']),
            Eq(field, 0, subdomain=subdomains['fsdomain']),
            Eq(field, 0, subdomain=subdomains['surface'])]


def operator_factor(pde, spatial, target):
    """
    The factor of the spatial operator `spatial` in the update of `target`
    solving `pde - spatial = 0`, which the transpose of the free surface of a
    model over a surface divides out (see `_odd_fold`). The update is linear in
    `spatial`, so it is solved with a symbol in its place.
    """
    operator = sympy.Dummy('spatial')
    return sympy.expand_mul(solve(pde - operator, target)).coeff(operator)


def adjoint_update(model, eq, pde, spatial):
    """
    The adjoint update `eq`, solving `pde - spatial = 0`, with the transpose of
    the free surface of a model over a surface: the update is computed in the
    mirror rows too, which the transpose folds onto the rows below.
    """
    return [eq, *freesurface(model, eq, operator_factor(pde, spatial, eq.lhs))]


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
    # Define time step to be updated
    unext = field.forward if forward else field.backward
    udt = field.dt if forward else field.dt.T
    # Get the spacial FD
    lap = laplacian(field, model, kernel)
    # Get source
    q = kwargs.get('q', 0)
    # Define PDE and update rule
    eq_time = solve(model.m * field.dt2 - lap - q + model.damp * udt, unext)

    if not forward and model.fs and 'surface' in model.grid.subdomains:
        return adjoint_update(model, Eq(unext, eq_time),
                              model.m*field.dt2 - q + model.damp*udt, lap)

    # Time-stepping stencil.
    eqns = [Eq(unext, eq_time, subdomain=model.grid.subdomains['physdomain'])]

    # Add free surface
    if model.fs:
        eqns.append(freesurface(model, Eq(unext, eq_time)))
    return eqns


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
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
                    name='Born', **kwargs)

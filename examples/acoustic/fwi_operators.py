from sympy import Eq
from sympy.abc import h, s

from devito import Operator, Forward, Backward, DenseData, TimeData, t, time
from examples.seismic import PointSource, Receiver


def ForwardOperator(model, data, source, time_order=2, space_order=4,
                    save=False, **kwargs):
    """
    Constructor method for the forward modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: None or IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    :param save : Saving flag, True saves all time steps, False only the three
    :param: u_ini : wavefield at the three first time step for non-zero initial condition
     required for the time marching scheme
    """
    m, damp = model.m, model.damp

    # Create the forward wavefield
    u = TimeData(name='u', shape=model.shape_domain, time_dim=data.shape[0],
                 time_order=time_order, space_order=space_order, save=save,
                 dtype=model.dtype)
    # Create source and receiver symbols
    src = PointSource(name='src', ntime=data.shape[0],
                      coordinates=source.receiver_coords)
    rec = Receiver(name='rec', ntime=data.shape[0],
                   coordinates=data.receiver_coords)

    # Derive stencil from symbolic equation
    if time_order == 2:
        laplacian = u.laplace
        biharmonic = 0
        # PDE for information
        # eqn = m * u.dt2 - laplacian + damp * u.dt
        dt = model.critical_dt
    else:
        laplacian = u.laplace
        biharmonic = u.laplace2(1/m)
        # PDE for information
        # eqn = m * u.dt2 - laplacian - s**2 / 12 * biharmonic + damp * u.dt
        dt = 1.73 * model.critical_dt

    # Derive both stencils from symbolic equation:
    # Create the stencil by hand instead of calling numpy solve for speed purposes
    # Simple linear solve of a u(t+dt) + b u(t) + c u(t-dt) = L for u(t+dt)
    stencil = 1 / (2 * m + s * damp) * (
        4 * m * u + (s * damp - 2 * m) * u.backward +
        2 * s**2 * (laplacian + s**2 / 12 * biharmonic))
    eqn = [Eq(u.forward, stencil)]

    # Construct expression to inject source values
    # Note that src and field terms have differing time indices:
    #   src[time, ...] - always accesses the "unrolled" time index
    #   u[ti + 1, ...] - accesses the forward stencil value
    ti = u.indices[0]
    src_term = src.inject(field=u, u_t=ti + 1, offset=model.nbpml,
                          expr=src * dt * dt / m, p_t=time)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u, u_t=ti, offset=model.nbpml)

    return Operator(stencils=eqn + src_term + rec_term,
                    subs={s: dt, h: model.get_spacing()},
                    time_axis=Forward, name='Forward', **kwargs)


def AdjointOperator(model, source, data, time_order=2, space_order=4, **kwargs):
    """
    Class to setup the adjoint modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: None or IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    m, damp = model.m, model.damp
    nt = data.shape[0]

    v = TimeData(name='v', shape=model.shape_domain, save=False,
                 time_order=time_order, space_order=space_order,
                 dtype=model.dtype)
    rec = Receiver(name='rec', ntime=nt,
                   coordinates=data.receiver_coords)
    srca = PointSource(name='srca', ntime=nt,
                       coordinates=source.receiver_coords)

    if time_order == 2:
        laplacian = v.laplace
        biharmonic = 0
        # PDE for information
        # eqn = m * u.dt2 - laplacian + damp * u.dt
        dt = model.critical_dt
    else:
        laplacian = v.laplace
        biharmonic = v.laplace2(1/m)
        # PDE for information
        # eqn = m * u.dt2 - laplacian - s**2 / 12 * biharmonic + damp * u.dt
        dt = 1.73 * model.critical_dt

    # Derive both stencils from symbolic equation
    stencil = 1 / (2 * m + s * damp) * (
        4 * m * v + (s * damp - 2 * m) * v.forward +
        2 * s**2 * (laplacian + s**2 / 12 * biharmonic))
    eqn = Eq(v.backward, stencil)

    # Construct expression to inject receiver values
    ti = v.indices[0]
    receivers = rec.inject(field=v, u_t=ti - 1, offset=model.nbpml,
                           expr=rec * dt * dt / m, p_t=time)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v, u_t=ti, offset=model.nbpml)

    return Operator(stencils=[eqn] + receivers + source_a,
                    subs={s: dt, h: model.get_spacing()},
                    time_axis=Backward, name='Adjoint', **kwargs)


def GradientOperator(model, source, data, time_order=2, space_order=4, **kwargs):
    """
    Class to setup the gradient operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    m, damp = model.m, model.damp
    nt = data.shape[0]

    # Gradient symbol
    grad = DenseData(name='grad', shape=model.shape_domain,
                     dtype=model.dtype)
    u = TimeData(name='u', shape=model.shape_domain, save=True,
                 time_dim=nt,
                 time_order=time_order, space_order=space_order,
                 dtype=model.dtype)
    v = TimeData(name='v', shape=model.shape_domain,save=False,
                 time_order=time_order, space_order=space_order,
                 dtype=model.dtype)
    rec = Receiver(name='rec', ntime=nt,
                   coordinates=data.receiver_coords)

    # Derive stencil from symbolic equation
    if time_order == 2:
        laplacian = v.laplace
        biharmonic = 0
        # PDE for information
        # eqn = m * v.dt2 - laplacian - damp * v.dt
        dt = model.critical_dt

        gradient_update = Eq(grad, grad - u.dt2 * v)
    else:
        laplacian = v.laplace
        biharmonic = v.laplace2(1/m)
        biharmonicu = - u.laplace2(1/(m**2))
        # PDE for information
        # eqn = m * v.dt2 - laplacian - s**2 / 12 * biharmonic + damp * v.dt
        dt = 1.73 * model.critical_dt
        gradient_update = Eq(grad, grad -
                             (u.dt2 -
                              s ** 2 / 12.0 * biharmonicu) * v)

    # Derive stencil from symbolic equation
    stencil = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * v + (s * damp - 2.0 * m) *
         v.forward + 2.0 * s ** 2 * (laplacian + s**2 / 12.0 * biharmonic))
    eqn = Eq(v.backward, stencil)

    # Add expression for receiver injection
    ti = v.indices[0]
    receivers = rec.inject(field=v, u_t=ti - 1, offset=model.nbpml,
                           expr=rec * dt * dt / m, p_t=time)

    return Operator(stencils=[eqn] + [gradient_update] + receivers,
                    subs={s: dt, h: model.get_spacing()},
                    time_axis=Backward, name='Gradient', **kwargs)


def BornOperator(model, source, data, time_order=2, space_order=4, **kwargs):
    """
    Class to setup the linearized modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: dmin : square slowness perturbation
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    m, damp = model.m, model.damp
    nt = data.shape[0]

    # Create source and receiver symbols
    src = PointSource(name='src', ntime=nt,
                      coordinates=source.receiver_coords)
    rec = Receiver(name='rec', ntime=data.shape[0],
                   coordinates=data.receiver_coords)

    # Create the forward wavefield
    u = TimeData(name="u", shape=model.shape_domain, time_dim=nt,
                 time_order=2, space_order=space_order,
                 dtype=model.dtype)
    U = TimeData(name="U", shape=model.shape_domain, time_dim=nt,
                 time_order=2, space_order=space_order,
                 dtype=model.dtype)
    dm = DenseData(name="dm", shape=model.shape_domain,
                   dtype=model.dtype)

    # Derive stencils from symbolic equation
    if time_order == 2:
        laplacianu = u.laplace
        biharmonicu = 0
        laplacianU = U.laplace
        biharmonicU = 0
        dt = model.critical_dt
    else:
        laplacianu = u.laplace
        biharmonicu = u.laplace2(1/m)
        laplacianU = U.laplace
        biharmonicU = U.laplace2(1/m)
        dt = 1.73 * model.critical_dt
        # first_eqn = m * u.dt2 - u.laplace + damp * u.dt
        # second_eqn = m * U.dt2 - U.laplace - dm* u.dt2 + damp * U.dt

    # Derive both stencils from symbolic equation
    stencil1 = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * u + (s * damp - 2.0 * m) *
         u.backward + 2.0 * s ** 2 * (laplacianu + s**2 / 12 * biharmonicu))
    stencil2 = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * U + (s * damp - 2.0 * m) *
         U.backward + 2.0 * s ** 2 * (laplacianU +
                                      s**2 / 12 * biharmonicU - dm * u.dt2))
    eqn1 = Eq(u.forward, stencil1)
    eqn2 = Eq(U.forward, stencil2)

    # Add source term expression for u
    ti = u.indices[0]
    source = src.inject(field=u, u_t=ti + 1, offset=model.nbpml,
                        expr=src * dt * dt / m, p_t=time)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U, u_t=ti, offset=model.nbpml)

    return Operator(stencils=[eqn1] + source + [eqn2] + receivers,
                    subs={s: dt, h: model.get_spacing()},
                    time_axis=Forward, name='Born', **kwargs)

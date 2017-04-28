from sympy import Eq, symbols

from devito.dimension import t, time
from devito.interfaces import Backward, Forward
from devito.stencilkernel import StencilKernel
from devito.operator import Operator


def ForwardOperator(model, u, src, rec, data, time_order=2, spc_order=6,
                    save=False, u_ini=None, legacy=True, **kwargs):
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
    nt = data.shape[0]
    s, h = symbols('s h')
    m, damp = model.m, model.damp
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

    # Create the stencil by hand instead of calling numpy solve for speed purposes
    # Simple linear solve of a u(t+dt) + b u(t) + c u(t-dt) = L for u(t+dt)
    stencil = 1 / (2 * m + s * damp) * (
        4 * m * u + (s * damp - 2 * m) * u.backward +
        2 * s**2 * (laplacian + s**2 / 12 * biharmonic))
    # Add substitutions for spacing (temporal and spatial)
    subs = {s: dt, h: model.get_spacing()}

    if legacy:
        kwargs.pop('dle', None)

        op = Operator(nt, model.shape_domain, stencils=Eq(u.forward, stencil), subs=subs,
                      spc_border=max(spc_order / 2, 2), time_order=2, forward=True,
                      dtype=model.dtype, **kwargs)

        # Insert source and receiver terms post-hoc
        op.input_params += [src, src.coordinates, rec, rec.coordinates]
        op.output_params += [rec]
        op.propagator.time_loop_stencils_a = src.add(m, u) + rec.read(u)
        op.propagator.add_devito_param(src)
        op.propagator.add_devito_param(src.coordinates)
        op.propagator.add_devito_param(rec)
        op.propagator.add_devito_param(rec.coordinates)

    else:
        dse = kwargs.get('dse', 'advanced')
        dle = kwargs.get('dle', 'advanced')

        # Create stencil expressions for operator, source and receivers
        eqn = Eq(u.forward, stencil)
        src_add = src.point2grid(u, m, u_t=u.indices[0] + 1, p_t=time)
        rec_read = Eq(rec, rec.grid2point(u, t=u.indices[0]))
        stencils = [eqn] + src_add + [rec_read]

        op = StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                           time_axis=Forward, name="Forward")

    return op


def AdjointOperator(model, v, srca, rec, data, time_order=2, spc_order=6,
                    save=False, u_ini=None, legacy=True, **kwargs):
    """
    Class to setup the adjoint modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: None or IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    nt = data.shape[0]
    s, h = symbols('s h')
    m, damp = model.m, model.damp
    # Derive stencil from symbolic equation
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

    # Create the stencil by hand instead of calling numpy solve for speed purposes
    # Simple linear solve of a u(t+dt) + b u(t) + c u(t-dt) = L for u(t+dt)
    stencil = 1 / (2 * m + s * damp) * (
        4 * m * v + (s * damp - 2 * m) * v.forward +
        2 * s**2 * (laplacian + s**2 / 12 * biharmonic))
    # Add substitutions for spacing (temporal and spatial)
    subs = {s: dt, h: model.get_spacing()}

    if legacy:
        kwargs.pop('dle', None)

        op = Operator(nt, model.shape_domain, stencils=Eq(v.backward, stencil), subs=subs,
                      spc_border=max(spc_order, 2), time_order=2, forward=False,
                      dtype=model.dtype, **kwargs)

        # Insert source and receiver terms post-hoc
        op.input_params += [srca, srca.coordinates, rec, rec.coordinates]
        op.output_params += [rec]
        op.propagator.time_loop_stencils_a = rec.add(m, v) + srca.read(v)
        op.propagator.add_devito_param(srca)
        op.propagator.add_devito_param(srca.coordinates)
        op.propagator.add_devito_param(rec)
        op.propagator.add_devito_param(rec.coordinates)

    else:
        dse = kwargs.get('dse', 'advanced')
        dle = kwargs.get('dle', 'advanced')

        # Create stencil expressions for operator, source and receivers
        eqn = Eq(v.backward, stencil)
        src_read = Eq(srca, srca.grid2point(v))
        rec_add = rec.point2grid(v, m, u_t=t - 1, p_t=time)
        stencils = [eqn] + rec_add + [src_read]

        op = StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                           time_axis=Backward, name="Adjoint")

    return op


def GradientOperator(model, v, grad, rec, u, data, time_order=2, spc_order=6,
                     legacy=True, **kwargs):
    """
    Class to setup the gradient operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    nt, nrec = data.shape
    s, h = symbols('s h')
    m, damp = model.m, model.damp

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

    # Create the stencil by hand instead of calling numpy solve for speed purposes
    # Simple linear solve of a v(t+dt) + b u(t) + c v(t-dt) = L for v(t-dt)
    stencil = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * v + (s * damp - 2.0 * m) *
         v.forward + 2.0 * s ** 2 * (laplacian + s**2 / 12.0 * biharmonic))
    # Add substitutions for spacing (temporal and spatial)
    subs = {s: dt, h: model.get_spacing()}
    # Add Gradient-specific updates. The dt2 is currently hacky
    #  as it has to match the cyclic indices

    if legacy:
        kwargs.pop('dle', None)
        gradient_update = gradient_update.subs(time, t)
        stencils = [gradient_update, Eq(v.backward, stencil)]
        op = Operator(rec.nt - 1, model.shape_domain,
                      stencils=stencils,
                      subs=[subs, subs, {}],
                      spc_border=max(spc_order, 2),
                      time_order=2,
                      forward=False,
                      dtype=model.dtype,
                      input_params=[m, v, damp, u, grad],
                      **kwargs)

        # Insert source and receiver terms post-hoc
        op.input_params += [rec, rec.coordinates]
        op.output_params += [grad]
        op.propagator.time_loop_stencils_b = rec.add(m, v, u_t=t + 1, p_t=t + 1)
        op.propagator.add_devito_param(rec)
        op.propagator.add_devito_param(rec.coordinates)

    else:
        dse = kwargs.get('dse', 'advanced')
        dle = kwargs.get('dle', 'advanced')

        # Create stencil expressions for operator, source and receivers
        eqn = Eq(v.backward, stencil)
        rec_add = rec.point2grid(v, m, u_t=t - 1, p_t=time)
        stencils = [eqn] + [gradient_update] + rec_add
        op = StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                           time_axis=Backward, name="Gradient")

    return op


def BornOperator(model, u, U, src, rec, dm, data, time_order=2, spc_order=6,
                 legacy=True, **kwargs):
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
    nt = data.shape[0]
    m, damp = model.m, model.damp
    s, h = symbols('s h')

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

    stencil1 = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * u + (s * damp - 2.0 * m) *
         u.backward + 2.0 * s ** 2 * (laplacianu + s**2 / 12 * biharmonicu))
    stencil2 = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * U + (s * damp - 2.0 * m) *
         U.backward + 2.0 * s ** 2 * (laplacianU +
                                      s**2 / 12 * biharmonicU - dm * u.dt2))
    # Add substitutions for spacing (temporal and spatial)
    subs = {s: dt, h: model.get_spacing()}
    if legacy:
        kwargs.pop('dle', None)
        stencils = [Eq(u.forward, stencil1), Eq(U.forward, stencil2)]
        op = Operator(nt, model.shape_domain,
                      stencils=stencils,
                      subs=[subs, subs],
                      spc_border=max(spc_order, 2),
                      time_order=2,
                      forward=True,
                      dtype=model.dtype,
                      **kwargs)

        # Insert source and receiver terms post-hoc
        op.input_params += [dm, src, src.coordinates, rec, rec.coordinates, U]
        op.output_params += [rec]
        op.propagator.time_loop_stencils_b = src.add(m, u, u_t=t - 1, p_t=t - 1)
        op.propagator.time_loop_stencils_a = rec.read(U)
        op.propagator.add_devito_param(dm)
        op.propagator.add_devito_param(U)
        op.propagator.add_devito_param(src)
        op.propagator.add_devito_param(src.coordinates)
        op.propagator.add_devito_param(rec)
        op.propagator.add_devito_param(rec.coordinates)

    else:
        dse = kwargs.get('dse', None)
        dle = kwargs.get('dle', None)

        # Create stencil expressions for operator, source and receivers
        eqn1 = [Eq(u.forward, stencil1)]
        eqn2 = [Eq(U.forward, stencil2)]
        src_add = src.point2grid(u, m, u_t=t + 1, p_t=time)
        rec_read = Eq(rec, rec.grid2point(U, t=U.indices[0]))
        stencils = eqn1 + src_add + eqn2 + [rec_read]

        op = StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                           time_axis=Forward, name="Born")

    return op

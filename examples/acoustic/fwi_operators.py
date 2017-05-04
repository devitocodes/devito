from sympy import Eq
from sympy.abc import h, s

from devito.dimension import t, time
from devito.interfaces import Backward, Forward
from devito.stencilkernel import StencilKernel


def ForwardOperator(model, u, src, rec, data, time_order=2, spc_order=6,
                    save=False, u_ini=None, **kwargs):
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

    dse = kwargs.get('dse', 'advanced')
    dle = kwargs.get('dle', 'advanced')

    # Create stencil expressions for operator, source and receivers
    eqn = Eq(u.forward, stencil)
    src_add = src.point2grid(u, m, u_t=u.indices[0] + 1, p_t=time)
    rec_read = Eq(rec, rec.grid2point(u, t=u.indices[0]))
    stencils = [eqn] + src_add + [rec_read]

    return StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                         time_axis=Forward, name="Forward")


def AdjointOperator(model, v, srca, rec, data, time_order=2, spc_order=6,
                    save=False, u_ini=None, **kwargs):
    """
    Class to setup the adjoint modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: None or IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
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

    dse = kwargs.get('dse', 'advanced')
    dle = kwargs.get('dle', 'advanced')

    # Create stencil expressions for operator, source and receivers
    eqn = Eq(v.backward, stencil)
    src_read = Eq(srca, srca.grid2point(v))
    rec_add = rec.point2grid(v, m, u_t=t - 1, p_t=time)
    stencils = [eqn] + rec_add + [src_read]

    return StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                         time_axis=Backward, name="Adjoint")


def GradientOperator(model, v, grad, rec, u, data, time_order=2, spc_order=6,
                     **kwargs):
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

    dse = kwargs.get('dse', 'advanced')
    dle = kwargs.get('dle', 'advanced')

    # Create stencil expressions for operator, source and receivers
    eqn = Eq(v.backward, stencil)
    rec_add = rec.point2grid(v, m, u_t=t - 1, p_t=time)
    stencils = [eqn] + [gradient_update] + rec_add
    return StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                         time_axis=Backward, name="Gradient")


def BornOperator(model, u, U, src, rec, dm, data, time_order=2, spc_order=6,
                 **kwargs):
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

    dse = kwargs.get('dse', None)
    dle = kwargs.get('dle', None)

    # Create stencil expressions for operator, source and receivers
    eqn1 = [Eq(u.forward, stencil1)]
    eqn2 = [Eq(U.forward, stencil2)]
    src_add = src.point2grid(u, m, u_t=t + 1, p_t=time)
    rec_read = Eq(rec, rec.grid2point(U, t=U.indices[0]))
    stencils = eqn1 + src_add + eqn2 + [rec_read]

    return StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                         time_axis=Forward, name="Born")

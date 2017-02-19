from sympy import Eq, solve, symbols

from devito.dimension import t, time
from devito.interfaces import DenseData, TimeData
from devito.operator import *
from devito.stencilkernel import StencilKernel
from examples.source_type import SourceLike


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

        op = Operator(nt, model.shape_pml, stencils=Eq(u.forward, stencil), subs=subs,
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
        compiler = kwargs.get('compiler', None)

        # Create stencil expressions for operator, source and receivers
        eqn = Eq(u.forward, stencil)
        src_add = src.point2grid(u, m, u_t=t, p_t=time)
        rec_read = Eq(rec, rec.grid2point(u))
        stencils = [eqn] + src_add + [rec_read]

        # TODO: The following time-index hackery is a legacy hangover
        # from the Operator/Propagator structure and is used here for
        # backward compatibiliy. We need re-examine this apporach carefully!

        # Shift time indices so that LHS writes into t only,
        # eg. u[t+2] = u[t+1] + u[t]  -> u[t] = u[t-1] + u[t-2]
        stencils = [e.subs(t, t + solve(e.lhs.args[0], t)[0])
                    if isinstance(e.lhs, TimeData) else e
                    for e in stencils]
        # Apply time substitutions as per legacy approach
        time_subs = {t + 2: t + 1, t: t + 2, t - 2: t, t - 1: t + 1, t + 1: t}
        subs.update(time_subs)

        op = StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                           compiler=compiler)

    return op


class AdjointOperator(Operator):
    """
    Class to setup the adjoint modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param src: None or IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    def __init__(self, model, data, src, recin, time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        s, h = symbols('s h')
        m, damp = model.m, model.damp
        v = TimeData(name="v", shape=model.shape_pml, time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=model.dtype)
        v.pad_time = False
        # Derive stencil from symbolic equation
        if time_order == 2:
            laplacian = v.laplace
            biharmonic = 0
            # PDE for information
            # eqn = m * v.dt2 - laplacian - damp * v.dt
            dt = model.critical_dt
        else:
            laplacian = v.laplace
            biharmonic = v.laplace2(1/m)
            # PDE for information
            # eqn = m * v.dt2 - laplacian - s**2 / 12 * biharmonic + damp * v.dt
            dt = 1.73 * model.critical_dt

        # Create the stencil by hand instead of calling numpy solve for speed purposes
        # Simple linear solve of a v(t+dt) + b u(t) + c v(t-dt) = L for v(t-dt)
        stencil = 1 / (2 * m + s * damp) * \
            (4 * m * v + (s * damp - 2 * m) *
             v.forward + 2 * s**2 * (laplacian + s ** 2 / 12.0 * biharmonic))

        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        # Source and receiver initialization
        srca = SourceLike(name="srca", npoint=src.traces.shape[1],
                          nt=nt, dt=dt, h=model.get_spacing(),
                          coordinates=src.receiver_coords,
                          ndim=len(model.shape), dtype=model.dtype,
                          nbpml=model.nbpml)
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(model.shape),
                         dtype=model.dtype, nbpml=model.nbpml)
        rec.data[:] = recin[:]

        super(AdjointOperator, self).__init__(nt, model.shape_pml,
                                              stencils=Eq(v.backward, stencil),
                                              subs=subs,
                                              spc_border=max(spc_order / 2, 2),
                                              time_order=2,
                                              forward=False,
                                              dtype=model.dtype,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [srca, srca.coordinates, rec, rec.coordinates]
        self.propagator.time_loop_stencils_a = rec.add(m, v) + srca.read(v)
        self.output_params = [srca]
        self.propagator.add_devito_param(srca)
        self.propagator.add_devito_param(srca.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class GradientOperator(Operator):
    """
    Class to setup the gradient operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    def __init__(self, model, data, recin, u, time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        s, h = symbols('s h')
        m, damp = model.m, model.damp
        v = TimeData(name="v", shape=model.shape_pml, time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=model.dtype)
        v.pad_time = False
        grad = DenseData(name="grad", shape=model.shape_pml, dtype=model.dtype)

        # Derive stencil from symbolic equation
        if time_order == 2:
            laplacian = v.laplace
            biharmonic = 0
            # PDE for information
            # eqn = m * v.dt2 - laplacian - damp * v.dt
            dt = model.critical_dt
            gradient_update = Eq(grad, grad - u.dt2 * v.forward)
        else:
            laplacian = v.laplace
            biharmonic = v.laplace2(1/m)
            biharmonicu = - u.laplace2(1/(m**2))
            # PDE for information
            # eqn = m * v.dt2 - laplacian - s**2 / 12 * biharmonic + damp * v.dt
            dt = 1.73 * model.critical_dt
            gradient_update = Eq(grad, grad -
                                 (u.dt2 -
                                  s ** 2 / 12.0 * biharmonicu) * v.forward)

        # Create the stencil by hand instead of calling numpy solve for speed purposes
        # Simple linear solve of a v(t+dt) + b u(t) + c v(t-dt) = L for v(t-dt)
        stencil = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * v + (s * damp - 2.0 * m) *
             v.forward + 2.0 * s ** 2 * (laplacian + s**2 / 12.0 * biharmonic))

        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        # Add Gradient-specific updates. The dt2 is currently hacky
        #  as it has to match the cyclic indices
        stencils = [gradient_update, Eq(v.backward, stencil)]

        # Receiver initialization
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(model.shape),
                         dtype=model.dtype, nbpml=model.nbpml)
        rec.data[:] = recin
        super(GradientOperator, self).__init__(rec.nt - 1, model.shape_pml,
                                               stencils=stencils,
                                               subs=[subs, subs, {}],
                                               spc_border=max(spc_order, 2),
                                               time_order=2,
                                               forward=False,
                                               dtype=model.dtype,
                                               input_params=[m, v, damp, u, grad],
                                               **kwargs)
        # Insert receiver term post-hoc
        self.input_params += [rec, rec.coordinates]
        self.output_params = [grad]
        self.propagator.time_loop_stencils_b = rec.add(m, v, u_t=t + 1, p_t=t + 1)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class BornOperator(Operator):
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
    def __init__(self, model, src, data, dmin, time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        nt, nsrc = src.shape
        m, damp = model.m, model.damp
        u = TimeData(name="u", shape=model.shape_pml, time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=model.dtype)
        U = TimeData(name="U", shape=model.shape_pml, time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=model.dtype)

        dm = DenseData(name="dm", shape=model.shape_pml, dtype=model.dtype)
        dm.data[:] = model.pad(dmin)
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
        # Add Born-specific updates and resets
        stencils = [Eq(u.forward, stencil1), Eq(U.forward, stencil2)]

        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(model.shape),
                         dtype=model.dtype, nbpml=model.nbpml)
        source = SourceLike(name="src", npoint=nsrc, nt=nt, dt=dt, h=model.get_spacing(),
                            coordinates=src.receiver_coords, ndim=len(model.shape),
                            dtype=model.dtype, nbpml=model.nbpml)
        source.data[:] = src.traces[:]

        super(BornOperator, self).__init__(nt, model.shape_pml,
                                           stencils=stencils,
                                           subs=[subs, subs],
                                           spc_border=max(spc_order, 2),
                                           time_order=2,
                                           forward=True,
                                           dtype=model.dtype,
                                           **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [dm, source, source.coordinates, rec, rec.coordinates, U]
        self.output_params = [rec]
        self.propagator.time_loop_stencils_b = source.add(m, u, u_t=t - 1, p_t=t - 1)
        self.propagator.time_loop_stencils_a = rec.read(U)
        self.propagator.add_devito_param(dm)
        self.propagator.add_devito_param(source)
        self.propagator.add_devito_param(source.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
        self.propagator.add_devito_param(U)

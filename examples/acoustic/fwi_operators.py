from sympy import Eq, solve, symbols

from devito.dimension import t, time
from devito.interfaces import Backward, DenseData, Forward, TimeData
from devito.operator import *
from devito.stencilkernel import StencilKernel
from examples.source_type import SourceLike


def ForwardOperator(model, u, src, rec, damp, data, time_order=2, spc_order=6,
                    save=False, u_ini=None, legacy=True, **kwargs):
    """
    Constructor method for the forward modelling operator in an acoustic media

    :param model: IGrid() object containing the physical parameters
    :param source: None or IShot() (not currently supported properly)
    :param damp: Dampening coeeficents for the ABCs
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    :param save : Saving flag, True saves all time steps, False only the three
    :param: u_ini : wavefield at the three first time step for non-zero initial condition
     required for the time marching scheme
    """
    nt = data.shape[0]
    B, s, h = symbols('B s h')
    m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
    m.data[:] = model.padm()
    # Derive stencil from symbolic equation
    if time_order == 2:
        laplacian = u.laplace
        biharmonic = 0
        # PDE for information
        # eqn = m * u.dt2 - laplacian + damp * u.dt
        dt = model.get_critical_dt()
    else:
        laplacian = u.laplace
        biharmonic = u.laplace2(1/m)
        # PDE for information
        # eqn = m * u.dt2 - laplacian - s**2 / 12 * biharmonic + damp * u.dt
        dt = 1.73 * model.get_critical_dt()

    # Create the stencil by hand instead of calling numpy solve for speed purposes
    eqn = m * u.dt2 + damp * u.dt + B
    stencil = solve(eqn, u.forward, rational=False, simplify=False, check=False)[0]
    stencil = stencil.xreplace({B: - laplacian - s**2 / 12 * biharmonic})
    # Add substitutions for spacing (temporal and spatial)
    subs = {s: dt, h: model.get_spacing()}

    if legacy:
        kwargs.pop('dle', None)

        op = Operator(nt, m.shape, stencils=Eq(u.forward, stencil), subs=subs,
                      spc_border=max(spc_order / 2, 2), time_order=2, forward=True,
                      dtype=m.dtype, **kwargs)

        # Insert source and receiver terms post-hoc
        op.input_params += [src, src.coordinates, rec, rec.coordinates]
        op.output_params += [rec]
        op.propagator.time_loop_stencils_a = src.add(m, u) + rec.read(u)
        op.propagator.add_devito_param(src)
        op.propagator.add_devito_param(src.coordinates)
        op.propagator.add_devito_param(rec)
        op.propagator.add_devito_param(rec.coordinates)

    else:
        dse = kwargs.get('dse', None)
        dle = kwargs.get('dle', None)
        compiler = kwargs.get('compiler', None)
        # Create stencil expressions for operator, source and receivers
        eqn = Eq(u.forward, stencil)
        src_add = src.point2grid(u, m, u_t=t + 1, p_t=time)
        rec_read = rec.grid2point(u, u_t=t + 1, p_t=time)
        stencils = [eqn] + src_add + [rec_read]

        op = StencilKernel(stencils=stencils, subs=subs, dse=dse, dle=dle,
                           compiler=compiler)

    return op


class AdjointOperator(Operator):
    """
    Class to setup the adjoint modelling operator in an acoustic media

    :param model: IGrid() object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param damp: Dampening coeeficents for the ABCs
    :param data: IShot() object containing the acquisition geometry and field data
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    def __init__(self, model, damp, data, src, recin,
                 time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        s, h, B = symbols('s h B')
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=damp.dtype, taxis=Backward)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        # Derive stencil from symbolic equation
        if time_order == 2:
            laplacian = v.laplace
            biharmonic = 0
            # PDE for information
            # eqn = m * v.dt2 - laplacian - damp * v.dt
            dt = model.get_critical_dt()
        else:
            laplacian = v.laplace
            biharmonic = v.laplace2(1/m)
            # PDE for information
            # eqn = m * v.dt2 - laplacian - s**2 / 12 * biharmonic + damp * v.dt
            dt = 1.73 * model.get_critical_dt()

        # Create the stencil by hand instead of calling numpy solve for speed purposes
        eqn = m * v.dt2 - damp * v.dt + B
        stencil = solve(eqn, v.backward, rational=False, simplify=False, check=False)[0]
        stencil = stencil.xreplace({B: - laplacian - s ** 2 / 12 * biharmonic})

        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        # Source and receiver initialization
        srca = SourceLike(name="srca", npoint=src.traces.shape[1],
                          nt=nt, dt=dt, h=model.get_spacing(),
                          coordinates=src.receiver_coords,
                          ndim=len(damp.shape), dtype=damp.dtype, nbpml=model.nbpml)
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin[:]

        super(AdjointOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(v.backward, stencil),
                                              subs=subs,
                                              spc_border=max(spc_order / 2, 2),
                                              time_order=2,
                                              forward=False,
                                              dtype=m.dtype,
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

    :param model: IGrid() object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param damp: Dampening coeeficents for the ABCs
    :param data: IShot() object containing the acquisition geometry and field data
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    def __init__(self, model, damp, data, recin, u, time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        s, h, B = symbols('s h B')
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=damp.dtype, taxis=Backward)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        grad = DenseData(name="grad", shape=m.shape, dtype=m.dtype)

        # Derive stencil from symbolic equation
        if time_order == 2:
            laplacian = v.laplace
            biharmonic = 0
            # PDE for information
            # eqn = m * v.dt2 - laplacian - damp * v.dt
            dt = model.get_critical_dt()
            gradient_update = Eq(grad, grad - u.dt2 * v.forward)
        else:
            laplacian = v.laplace
            biharmonic = v.laplace2(1/m)
            biharmonicu = - u.laplace2(1/(m**2))
            # PDE for information
            # eqn = m * v.dt2 - laplacian - s**2 / 12 * biharmonic + damp * v.dt
            dt = 1.73 * model.get_critical_dt()
            gradient_update = Eq(grad, grad -
                                 (u.dt2 -
                                  s ** 2 / 12.0 * biharmonicu) * v.forward)

        # Create the stencil by hand instead of calling numpy solve for speed purposes
        # Simple linear solve of a v(t+dt) + b u(t) + c v(t-dt) = L for v(t-dt)
        eqn = m * v.dt2 - damp * v.dt + B
        stencil = solve(eqn, v.backward, rational=False, simplify=False, check=False)[0]
        stencil = stencil.xreplace({B: - laplacian - s ** 2 / 12 * biharmonic})
        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        # Add Gradient-specific updates. The dt2 is currently hacky
        #  as it has to match the cyclic indices
        stencils = [Eq(v.backward, stencil), gradient_update]

        # Receiver initialization
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin
        super(GradientOperator, self).__init__(rec.nt - 1, m.shape,
                                               stencils=stencils,
                                               subs=[subs, subs, {}],
                                               spc_border=max(spc_order, 2),
                                               time_order=2,
                                               forward=False,
                                               dtype=m.dtype,
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

    :param model: IGrid() object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param damp: Dampening coeeficents for the ABCs
    :param data: IShot() object containing the acquisition geometry and field data
    :param: dmin : square slowness perturbation
    :param: recin : receiver data for the adjoint source
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    """
    def __init__(self, model, src, damp, data, dmin, time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.shape
        nt, nsrc = src.shape
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=damp.dtype, taxis=Forward)
        U = TimeData(name="U", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=damp.dtype, taxis=Forward)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()

        dm = DenseData(name="dm", shape=model.get_shape_comp(), dtype=damp.dtype)
        dm.data[:] = model.pad(dmin)
        s, h, B, C = symbols('s h B C')

        # Derive stencils from symbolic equation
        if time_order == 2:
            biharmonicu = 0
            biharmonicU = 0
            dt = model.get_critical_dt()
        else:
            biharmonicu = u.laplace2(1/m)
            biharmonicU = U.laplace2(1/m)
            dt = 1.73 * model.get_critical_dt()
            # first_eqn = m * u.dt2 - u.laplace + damp * u.dt
            # second_eqn = m * U.dt2 - U.laplace - dm* u.dt2 + damp * U.dt

        first_eqn = m * u.dt2 + damp * u.dt + B
        second_eqn = m * U.dt2 + damp * U.dt + C
        stencil1 = solve(first_eqn, u.forward, rational=False, simplify=False, check=False)[0]
        stencil2 = solve(second_eqn, U.forward, rational=False, simplify=False, check=False)[0]
        stencil1 = stencil1.xreplace({B: - u.laplace - s**2 / 12 * biharmonicu})
        stencil2 = stencil2.xreplace({C: - U.laplace - dm * u.dt2 -
                                     s**2 / 12 * biharmonicU})
        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        # Add Born-specific updates and resets
        stencils = [Eq(u.forward, stencil1), Eq(U.forward, stencil2)]

        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        source = SourceLike(name="src", npoint=nsrc, nt=nt, dt=dt, h=model.get_spacing(),
                            coordinates=src.receiver_coords, ndim=len(damp.shape),
                            dtype=damp.dtype, nbpml=model.nbpml)
        source.data[:] = src.traces[:]

        super(BornOperator, self).__init__(nt, m.shape,
                                           stencils=stencils,
                                           subs=[subs, subs],
                                           spc_border=max(spc_order, 2),
                                           time_order=2,
                                           forward=True,
                                           dtype=m.dtype,
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

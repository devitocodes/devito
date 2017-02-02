from sympy import Eq, symbols

from devito.dimension import t
from devito.interfaces import DenseData, TimeData
from devito.operator import *
from examples.source_type import SourceLike


class ForwardOperator(Operator):
    """
    Class to setup the forward modelling operator in an acoustic media

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
    def __init__(self, model, source, damp, data, time_order=2, spc_order=6,
                 save=False, u_ini=None, **kwargs):
        nt, nrec = data.shape
        nt, nsrc = source.shape
        s, h = symbols('s h')
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order, save=save,
                     dtype=damp.dtype)
        if u_ini is not None:
            u.data[0:3, :] = u_ini[:]
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        u.pad_time = save
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
        # Simple linear solve of a u(t+dt) + b u(t) + c u(t-dt) = L for u(t+dt)
        stencil = 1 / (2 * m + s * damp) * (
            4 * m * u + (s * damp - 2 * m) * u.backward +
            2 * s**2 * (laplacian + s**2 / 12 * biharmonic))
        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        # Receiver initialization
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        src = SourceLike(name="src", npoint=nsrc, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=source.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        src.data[:] = source.traces[:]

        super(ForwardOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(u.forward, stencil),
                                              subs=subs,
                                              spc_border=max(spc_order, 2),
                                              time_order=2,
                                              forward=True,
                                              dtype=m.dtype,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [src, src.coordinates, rec, rec.coordinates]
        self.output_params += [rec]
        self.propagator.time_loop_stencils_a = src.add(m, u) + rec.read(u)
        self.propagator.add_devito_param(src)
        self.propagator.add_devito_param(src.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


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
        s, h = symbols('s h')
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=damp.dtype)
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
                          ndim=len(damp.shape), dtype=damp.dtype, nbpml=model.nbpml)
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin[:]

        super(AdjointOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(v.backward, stencil),
                                              subs=subs,
                                              spc_border=max(spc_order, 2),
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
        s, h = symbols('s h')
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=damp.dtype)
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
                                               input_params=[m, v, damp, u],
                                               **kwargs)
        # Insert receiver term post-hoc
        self.input_params += [grad, rec, rec.coordinates]
        self.output_params = [grad]
        self.propagator.time_loop_stencils_b = rec.add(m, v, t + 1)
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
                     save=False, dtype=damp.dtype)
        U = TimeData(name="U", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()

        dm = DenseData(name="dm", shape=model.get_shape_comp(), dtype=damp.dtype)
        dm.data[:] = model.pad(dmin)
        s, h = symbols('s h')

        # Derive stencils from symbolic equation
        if time_order == 2:
            laplacianu = u.laplace
            biharmonicu = 0
            laplacianU = u.laplace
            biharmonicU = 0
            dt = model.get_critical_dt()
        else:
            laplacianu = u.laplace
            biharmonicu = U.laplace2(1/m)
            laplacianU = u.laplace
            biharmonicU = U.laplace2(1/m)
            dt = 1.73 * model.get_critical_dt()
            # first_eqn = m * u.dt2 - u.laplace + damp * u.dt
            # second_eqn = m * U.dt2 - U.laplace - dm* u.dt2 + damp * U.dt

        stencil1 = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * u + (s * damp - 2.0 * m) *
             u.backward + 2.0 * s ** 2 * (laplacianu + s**2 / 12 * biharmonicu))
        stencil2 = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * u + (s * damp - 2.0 * m) *
             u.backward + 2.0 * s ** 2 * (laplacianU +
                                          s**2 / 12 * biharmonicU - dm * u.dt2))
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
        self.propagator.time_loop_stencils_b = source.add(m, u, t - 1)
        self.propagator.time_loop_stencils_a = rec.read(U)
        self.propagator.add_devito_param(dm)
        self.propagator.add_devito_param(source)
        self.propagator.add_devito_param(source.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
        self.propagator.add_devito_param(U)

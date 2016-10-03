import numpy as np
from sympy import simplify, factor, Eq, solve, symbols

from devito.dimension import t
from devito.interfaces import DenseData, TimeData
from devito.operator import *
from examples.source_type import SourceLike


class ForwardOperator(Operator):
    def __init__(self, model, src, damp, data, time_order=2, spc_order=6,
                 save=False, **kwargs):
        nrec, nt = data.shape
        s, h = symbols('s h')
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order, save=save,
                     dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        u.pad_time = save
        # Derive stencil from symbolic equation
        if time_order == 2:
            Lap = u.laplace
            Lap2 = 0
            # PDE for information
            # eqn = m * u.dt2 - Lap + damp * u.dt
            dt = model.get_critical_dt()
        else:
            Lap = u.laplace
            Lap2 = 1 / m * u.laplace2
            # PDE for information
            # eqn = m * u.dt2 - Lap - s**2 / 12 * Lap2 + damp * u.dt
            dt = 1.73 * model.get_critical_dt()

        # Create the stencil by hand instead of calling numpy solve for speed purposes
        # Simple linear solve of a u(t+dt) + b u(t) + c u(t-dt) = L for u(t+dt)
        stencil = 1 / (2 * m + s * damp) * \
            (4 * m * u + (s * damp - 2 * m) *
             u.backward + 2 * s ** 2 * (Lap + s**2 / 12 * Lap2))
        # eqn = m * u.dt2 - Lap - Lap2 + damp* u.dt
        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        stencil = simplify(stencil)
        # Receiver initialization
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)

        super(ForwardOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(u.forward, stencil),
                                              subs=subs,
                                              spc_border=max(spc_order/2, 2),
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
    def __init__(self, model, damp, data, recin, time_order=2, spc_order=6, **kwargs):
        nrec, nt = data.shape
        s, h = symbols('s h')
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=2, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        # Derive stencil from symbolic equation
        if time_order == 2:
            Lap = v.laplace
            Lap2 = 0
            # PDE for information
            # eqn = m * v.dt2 - Lap - damp * v.dt
            dt = model.get_critical_dt()
        else:
            Lap = v.laplace
            Lap2 = 1 / m * v.laplace2
            # PDE for information
            # eqn = m * v.dt2 - Lap - s**2 / 12 * Lap2 + damp * v.dt
            dt = 1.73 * model.get_critical_dt()

        # Create the stencil by hand instead of calling numpy solve for speed purposes
        # Simple linear solve of a v(t+dt) + b u(t) + c v(t-dt) = L for v(t-dt)
        stencil = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * v + (s * damp - 2.0 * m) *
             v.forward + 2.0 * s ** 2 * (Lap + s ** 2 / 12.0 * Lap2))

        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        # Source and receiver initialization
        srca = SourceLike(name="srca", npoint=1, nt=nt, dt=dt, h=model.get_spacing(),
                          coordinates=np.array(data.source_coords,
                                               dtype=damp.dtype)[np.newaxis, :],
                          ndim=len(damp.shape), dtype=damp.dtype, nbpml=model.nbpml)
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin[:]

        super(AdjointOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(v.backward, stencil),
                                              subs=subs,
                                              spc_border=max(spc_order/2, 2),
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
    def __init__(self, model, damp, data, recin, u, time_order=2, spc_order=6, **kwargs):
        nrec, nt = data.shape
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
            Lap = v.laplace
            Lap2 = 0
            # PDE for information
            # eqn = m * v.dt2 - Lap - damp * v.dt
            dt = model.get_critical_dt()
            gradient_update = Eq(grad, grad - u.dt2 * v.forward)
        else:
            Lap = v.laplace
            Lap2 = 1 / m * v.laplace2
            Lap2u = - 1 / m**2 * u.laplace2
            # PDE for information
            # eqn = m * v.dt2 - Lap - s**2 / 12 * Lap2 + damp * v.dt
            dt = 1.73 * model.get_critical_dt()
            gradient_update = Eq(grad, grad -
                                 (u.dt2 -
                                  s ** 2 / 12.0 * Lap2u) * v.forward)

        # Create the stencil by hand instead of calling numpy solve for speed purposes
        # Simple linear solve of a v(t+dt) + b u(t) + c v(t-dt) = L for v(t-dt)
        stencil = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * v + (s * damp - 2.0 * m) *
             v.forward + 2.0 * s ** 2 * (Lap + s**2 / 12.0 * Lap2))

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
                                               spc_border=max(spc_order/2, 2),
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
    def __init__(self, model, src, damp, data, dmin, time_order=2, spc_order=6, **kwargs):
        nrec, nt = data.shape
        s, h = symbols('s h')
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

        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)

        # Derive stencils from symbolic equation
        if time_order == 2:
            Lapu = u.laplace
            Lap2u = 0
            LapU = u.laplace
            Lap2U = 0
            dt = model.get_critical_dt()
        else:
            Lapu = u.laplace
            Lap2u = 1 / m * U.laplace2
            LapU = u.laplace
            Lap2U = 1 / m * U.laplace2
            dt = 1.73 * model.get_critical_dt()
            # first_eqn = m * u.dt2 - u.laplace - damp * u.dt
            # second_eqn = m * U.dt2 - U.laplace - damp * U.dt

        stencil1 = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * u + (s * damp - 2.0 * m) *
             u.backward + 2.0 * s ** 2 * (Lapu + s**2 / 12 * Lap2u))
        src2 = -(dt ** -2) * (- 2 * u + u.forward + u.backward) * dm
        stencil2 = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * u + (s * damp - 2.0 * m) *
             u.backward + 2.0 * s ** 2 * (LapU + s**2 / 12 * Lap2U + src2))
        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: src.h}
        # Add Born-specific updates and resets
        insert_second_source = Eq(U, U + (dt * dt) / m*src2)
        stencils = [Eq(u.forward, stencil1), Eq(U.forward, stencil2),
                    insert_second_source]
        super(BornOperator, self).__init__(src.nt, m.shape,
                                           stencils=stencils,
                                           subs=[subs, subs, {}, {}],
                                           spc_border=max(spc_order/2, 2),
                                           time_order=2,
                                           forward=True,
                                           dtype=m.dtype,
                                           **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [dm, src, src.coordinates, rec, rec.coordinates, U]
        self.output_params = [rec]
        self.propagator.time_loop_stencils_b = src.add(m, u, t - 1)
        self.propagator.time_loop_stencils_a = rec.read(U)
        self.propagator.add_devito_param(dm)
        self.propagator.add_devito_param(src)
        self.propagator.add_devito_param(src.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
        self.propagator.add_devito_param(U)

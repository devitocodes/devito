import numpy as np
from sympy import Eq, solve, symbols

from devito.dimension import t
from devito.interfaces import DenseData, TimeData
from devito.operator import *
from examples.source_type import SourceLike


class ForwardOperator(Operator):
    def __init__(self, model, src, damp, data, time_order=2, spc_order=6,
                 save=False, **kwargs):
        nrec, nt = data.shape
        dt = model.get_critical_dt()
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order, save=save,
                     dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        u.pad_time = save
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        # Derive stencil from symbolic equation
        eqn = m * u.dt2 - u.laplace + damp * u.dt
        stencil = solve(eqn, u.forward)[0]
        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: dt, h: model.get_spacing()}
        super(ForwardOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(u.forward, stencil),
                                              subs=subs,
                                              spc_border=spc_order/2,
                                              time_order=time_order,
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
        dt = model.get_critical_dt()
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        srca = SourceLike(name="srca", npoint=1, nt=nt, dt=dt, h=model.get_spacing(),
                          coordinates=np.array(data.source_coords,
                                               dtype=damp.dtype)[np.newaxis, :],
                          ndim=len(damp.shape), dtype=damp.dtype, nbpml=model.nbpml)
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin[:]
        # Derive stencil from symbolic equation
        eqn = m * v.dt2 - v.laplace - damp * v.dt
        stencil = solve(eqn, v.backward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: model.get_critical_dt(), h: model.get_spacing()}
        super(AdjointOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(v.backward, stencil),
                                              subs=subs,
                                              spc_border=spc_order/2,
                                              time_order=time_order,
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
        dt = model.get_critical_dt()
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin
        grad = DenseData(name="grad", shape=m.shape, dtype=m.dtype)

        # Derive stencil from symbolic equation
        eqn = m * v.dt2 - v.laplace - damp * v.dt
        stencil = solve(eqn, v.backward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: model.get_critical_dt(), h: model.get_spacing()}
        # Add Gradient-specific updates. The dt2 is currently hacky
        #  as it has to match the cyclic indices
        gradient_update = Eq(grad, grad - s**-2*(v + v.forward - 2 * v.forward.forward) *
                             u.forward)
        stencils = [gradient_update, Eq(v.backward, stencil)]
        super(GradientOperator, self).__init__(rec.nt - 1, m.shape,
                                               stencils=stencils,
                                               subs=[subs, subs, {}],
                                               spc_border=spc_order/2,
                                               time_order=time_order,
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
        dt = model.get_critical_dt()
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        U = TimeData(name="U", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()

        dm = DenseData(name="dm", shape=model.get_shape_comp(), dtype=damp.dtype)
        dm.data[:] = model.pad(dmin)

        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)

        # Derive stencils from symbolic equation
        first_eqn = m * u.dt2 - u.laplace - damp * u.dt
        first_stencil = solve(first_eqn, u.forward)[0]
        second_eqn = m * U.dt2 - U.laplace - damp * U.dt
        second_stencil = solve(second_eqn, U.forward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: src.dt, h: src.h}

        # Add Born-specific updates and resets
        src2 = -(src.dt**-2) * (- 2 * u + u.forward + u.backward) * dm
        insert_second_source = Eq(U, U + (src.dt * src.dt) / m*src2)
        stencils = [Eq(u.forward, first_stencil), Eq(U.forward, second_stencil),
                    insert_second_source]
        super(BornOperator, self).__init__(src.nt, m.shape,
                                           stencils=stencils,
                                           subs=[subs, subs, {}, {}],
                                           spc_border=spc_order/2,
                                           time_order=time_order,
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

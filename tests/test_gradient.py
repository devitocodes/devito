
import numpy as np
import pytest
from numpy import linalg

from devito import Function, info, TimeFunction, Operator, Eq
from examples.seismic.acoustic.acoustic_example import smooth, acoustic_setup as setup
from examples.seismic.acoustic.operators import iso_stencil
from examples.seismic import Receiver, demo_model, setup_geometry


class TestGradient(object):

    @pytest.mark.parametrize('space_order', [4])
    @pytest.mark.parametrize('kernel', ['OT2'])
    @pytest.mark.parametrize('shape', [(70, 80)])
    def test_gradient_checkpointing(self, shape, kernel, space_order):
        r"""
        This test ensures that the FWI gradient computed with devito
        satisfies the Taylor expansion property:
        .. math::
            \Phi(m0 + h dm) = \Phi(m0) + \O(h) \\
            \Phi(m0 + h dm) = \Phi(m0) + h \nabla \Phi(m0) + \O(h^2) \\
            \Phi(m0) = .5* || F(m0 + h dm) - D ||_2^2

        where
        .. math::
            \nabla \Phi(m0) = <J^T \delta d, dm> \\
            \delta d = F(m0+ h dm) - D \\

        with F the Forward modelling operator.
        """
        spacing = tuple(10. for _ in shape)
        wave = setup(shape=shape, spacing=spacing, dtype=np.float64,
                     kernel=kernel, space_order=space_order,
                     nbl=40)

        v0 = Function(name='v0', grid=wave.model.grid, space_order=space_order)
        smooth(v0, wave.model.vp)

        # Compute receiver data for the true velocity
        rec, u, _ = wave.forward()

        # Compute receiver data and full wavefield for the smooth velocity
        rec0, u0, _ = wave.forward(vp=v0, save=True)

        # Gradient: <J^T \delta d, dm>
        residual = Receiver(name='rec', grid=wave.model.grid, data=rec0.data - rec.data,
                            time_range=wave.geometry.time_axis,
                            coordinates=wave.geometry.rec_positions)

        gradient, _ = wave.jacobian_adjoint(residual, u0, vp=v0, checkpointing=True)
        gradient2, _ = wave.jacobian_adjoint(residual, u0, vp=v0, checkpointing=False)
        assert np.allclose(gradient.data, gradient2.data)

    @pytest.mark.parametrize('tn', [750.])
    @pytest.mark.parametrize('spacing', [(10, 10)])
    @pytest.mark.parametrize('dtype', [np.float32])
    @pytest.mark.parametrize('nbl', [40])
    @pytest.mark.parametrize('preset', ['layers-isotropic'])
    @pytest.mark.parametrize('space_order', [4])
    @pytest.mark.parametrize('kernel', ['OT2'])
    @pytest.mark.parametrize('shape', [(101, 101)])
    def test_gradient_equivalence(self, shape, kernel, space_order, preset, nbl, dtype,
                                  spacing, tn):
        model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                           dtype=dtype, spacing=spacing)
        m = model.m
        geometry = setup_geometry(model, tn)
        dt = model.critical_dt
        src = geometry.src
        rec = geometry.rec
        s = model.grid.stepping_dim.spacing
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order,
                         save=geometry.nt)

        eqn_fwd = iso_stencil(u, model, kernel)
        src_term = src.inject(field=u.forward, expr=src * s**2 / m)
        rec_term = rec.interpolate(expr=u)

        fwd_op = Operator(eqn_fwd + src_term + rec_term, subs=model.spacing_map,
                          name='Forward')

        grad_u = Function(name='gradu', grid=model.grid)
        grad_v = Function(name='gradv', grid=model.grid)
        grad_uv = Function(name='graduv', grid=model.grid)
        v = TimeFunction(name='v', grid=model.grid, save=None, time_order=2,
                         space_order=space_order)

        eqn_adj = iso_stencil(v, model, kernel, forward=False)
        receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

        gradient_update_v = Eq(grad_v, grad_v - u * v.dt2)
        grad_op_v = Operator(eqn_adj + receivers + [gradient_update_v],
                             subs=model.spacing_map, name='GradientV')

        gradient_update_u = Eq(grad_u, grad_u - u.dt2 * v)
        grad_op_u = Operator(eqn_adj + receivers + [gradient_update_u],
                             subs=model.spacing_map, name='GradientU')

        gradient_update_uv = Eq(grad_uv, grad_uv + u.dt * v.dt)
        grad_op_uv = Operator(eqn_adj + receivers + [gradient_update_uv],
                              subs=model.spacing_map, name='GradientUV')

        fwd_op.apply(dt=dt)

        grad_op_u.apply(dt=dt)

        # Reset v before calling the second operator since the object is shared
        v.data[:] = 0.
        grad_op_v.apply(dt=dt)

        v.data[:] = 0.
        grad_op_uv.apply(dt=dt)

        assert(np.allclose(grad_u.data, grad_v.data, rtol=1e-3, atol=1e-3))
        assert(np.allclose(grad_u.data, grad_uv.data, rtol=1e-3, atol=1e-3))

    @pytest.mark.parametrize('space_order', [4])
    @pytest.mark.parametrize('kernel', ['OT2'])
    @pytest.mark.parametrize('shape', [(70, 80)])
    @pytest.mark.parametrize('checkpointing', [True, False])
    def test_gradientFWI(self, shape, kernel, space_order, checkpointing):
        r"""
        This test ensures that the FWI gradient computed with devito
        satisfies the Taylor expansion property:
        .. math::
            \Phi(m0 + h dm) = \Phi(m0) + \O(h) \\
            \Phi(m0 + h dm) = \Phi(m0) + h \nabla \Phi(m0) + \O(h^2) \\
            \Phi(m0) = .5* || F(m0 + h dm) - D ||_2^2

        where
        .. math::
            \nabla \Phi(m0) = <J^T \delta d, dm> \\
            \delta d = F(m0+ h dm) - D \\

        with F the Forward modelling operator.
        """
        spacing = tuple(10. for _ in shape)
        wave = setup(shape=shape, spacing=spacing, dtype=np.float64,
                     kernel=kernel, space_order=space_order,
                     nbl=40)

        v0 = Function(name='v0', grid=wave.model.grid, space_order=space_order)
        smooth(v0, wave.model.vp)
        v = wave.model.vp.data
        dm = np.float64(wave.model.vp.data**(-2) - v0.data**(-2))
        # Compute receiver data for the true velocity
        rec, u, _ = wave.forward()

        # Compute receiver data and full wavefield for the smooth velocity
        rec0, u0, _ = wave.forward(vp=v0, save=True)

        # Objective function value
        F0 = .5*linalg.norm(rec0.data - rec.data)**2

        # Gradient: <J^T \delta d, dm>
        residual = Receiver(name='rec', grid=wave.model.grid, data=rec0.data - rec.data,
                            time_range=wave.geometry.time_axis,
                            coordinates=wave.geometry.rec_positions)

        gradient, _ = wave.jacobian_adjoint(residual, u0, vp=v0,
                                            checkpointing=checkpointing)
        G = np.dot(gradient.data.reshape(-1), dm.reshape(-1))

        # FWI Gradient test
        H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
        error1 = np.zeros(7)
        error2 = np.zeros(7)
        for i in range(0, 7):
            # Add the perturbation to the model
            def initializer(data):
                data[:] = np.sqrt(v0.data**2 * v**2 /
                                  ((1 - H[i]) * v**2 + H[i] * v0.data**2))
            vloc = Function(name='vloc', grid=wave.model.grid, space_order=space_order,
                            initializer=initializer)
            # Data for the new model
            d = wave.forward(vp=vloc)[0]
            # First order error Phi(m0+dm) - Phi(m0)
            F_i = .5*linalg.norm((d.data - rec.data).reshape(-1))**2
            error1[i] = np.absolute(F_i - F0)
            # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
            error2[i] = np.absolute(F_i - F0 - H[i] * G)

        # Test slope of the  tests
        p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
        p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
        info('1st order error, Phi(m0+dm)-Phi(m0): %s' % (p1))
        info(r'2nd order error, Phi(m0+dm)-Phi(m0) - <J(m0)^T \delta d, dm>: %s' % (p2))
        assert np.isclose(p1[0], 1.0, rtol=0.1)
        assert np.isclose(p2[0], 2.0, rtol=0.1)

    @pytest.mark.parametrize('space_order', [4])
    @pytest.mark.parametrize('kernel', ['OT2'])
    @pytest.mark.parametrize('shape', [(70, 80)])
    def test_gradientJ(self, shape, kernel, space_order):
        r"""
        This test ensures that the Jacobian computed with devito
        satisfies the Taylor expansion property:
        .. math::
            F(m0 + h dm) = F(m0) + \O(h) \\
            F(m0 + h dm) = F(m0) + J dm + \O(h^2) \\

        with F the Forward modelling operator.
        """
        spacing = tuple(15. for _ in shape)
        wave = setup(shape=shape, spacing=spacing, dtype=np.float64,
                     kernel=kernel, space_order=space_order,
                     tn=1000., nbl=10+space_order/2)

        v0 = Function(name='v0', grid=wave.model.grid, space_order=space_order)
        smooth(v0, wave.model.vp)
        v = wave.model.vp.data
        dm = np.float64(wave.model.vp.data**(-2) - v0.data**(-2))
        linrec = Receiver(name='rec', grid=wave.model.grid,
                          time_range=wave.geometry.time_axis,
                          coordinates=wave.geometry.rec_positions)

        # Compute receiver data and full wavefield for the smooth velocity
        rec, u0, _ = wave.forward(vp=v0, save=False)

        # Gradient: J dm
        Jdm, _, _, _ = wave.jacobian(dm, rec=linrec, vp=v0)
        # FWI Gradient test
        H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
        error1 = np.zeros(7)
        error2 = np.zeros(7)
        for i in range(0, 7):
            # Add the perturbation to the model
            def initializer(data):
                data[:] = np.sqrt(v0.data**2 * v**2 /
                                  ((1 - H[i]) * v**2 + H[i] * v0.data**2))
            vloc = Function(name='vloc', grid=wave.model.grid, space_order=space_order,
                            initializer=initializer)
            # Data for the new model
            d = wave.forward(vp=vloc)[0]
            delta_d = (d.data - rec.data).reshape(-1)
            # First order error F(m0 + hdm) - F(m0)

            error1[i] = np.linalg.norm(delta_d, 1)
            # Second order term F(m0 + hdm) - F(m0) - J dm
            error2[i] = np.linalg.norm(delta_d - H[i] * Jdm.data.reshape(-1), 1)

        # Test slope of the  tests
        p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
        p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
        info('1st order error, Phi(m0+dm)-Phi(m0) with slope: %s compared to 1' % (p1[0]))
        info(r'2nd order error, Phi(m0+dm)-Phi(m0) - <J(m0)^T \delta d, dm>with slope:'
             ' %s comapred to 2' % (p2[0]))
        assert np.isclose(p1[0], 1.0, rtol=0.1)
        assert np.isclose(p2[0], 2.0, rtol=0.1)


if __name__ == "__main__":
    TestGradient().test_gradientFWI(shape=(70, 80), kernel='OT2', space_order=4,
                                    checkpointing=False)

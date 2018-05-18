import numpy as np
import pytest
from numpy import linalg
from conftest import skipif_yask, unit_box, points

from devito import clear_cache, Operator
from devito.logger import info
from examples.seismic import demo_model, TimeAxis, RickerSource, Receiver
from examples.seismic.acoustic import AcousticWaveSolver


presets = {
    'constant': {'preset': 'constant-isotropic'},
    'layers': {'preset': 'layers-isotropic', 'ratio': 3},
}


@skipif_yask
class TestAdjoint(object):

    def setup_method(self, method):
        # Some of these tests are memory intensive as it requires to store the entire
        # forward wavefield to compute the gradient (nx.ny.nz.nt). We therefore call
        # 'clear_cache()' to release any remaining memory from the previous tests or
        # previous instances (different parametrizations) of these tests
        clear_cache()

    @pytest.mark.parametrize('mkey, shape, kernel, space_order, nbpml', [
        # 1 tests with varying time and space orders
        ('layers', (60, ), 'OT2', 4, 10), ('layers', (60, ), 'OT2', 8, 10),
        ('layers', (60, ), 'OT4', 4, 10), ('layers', (60, ), 'OT4', 8, 10),
        # 2D tests with varying time and space orders
        ('layers', (60, 70), 'OT2', 4, 10), ('layers', (60, 70), 'OT2', 8, 10),
        ('layers', (60, 70), 'OT2', 12, 10), ('layers', (60, 70), 'OT4', 4, 10),
        ('layers', (60, 70), 'OT4', 8, 10), ('layers', (60, 70), 'OT4', 12, 10),
        # 3D tests with varying time and space orders
        ('layers', (60, 70, 80), 'OT2', 4, 10), ('layers', (60, 70, 80), 'OT2', 8, 10),
        ('layers', (60, 70, 80), 'OT2', 12, 10), ('layers', (60, 70, 80), 'OT4', 4, 10),
        ('layers', (60, 70, 80), 'OT4', 8, 10), ('layers', (60, 70, 80), 'OT4', 12, 10),
        # Constant model in 2D and 3D
        ('constant', (60, 70), 'OT2', 8, 14), ('constant', (60, 70, 80), 'OT2', 8, 14),
    ])
    def test_adjoint_F(self, mkey, shape, kernel, space_order, nbpml):
        """
        Adjoint test for the forward modeling operator.
        The forward modeling operator F generates a shot record (measurements)
        from a source while the adjoint of F generates measurments at the source
        location from data. This test uses the conventional dot test:
        < Fx, y> = <x, F^T y>
        """
        t0 = 0.0  # Start time
        tn = 500.  # Final time
        nrec = 130  # Number of receivers

        # Create model from preset
        model = demo_model(spacing=[15. for _ in shape], dtype=np.float64,
                           space_order=space_order, shape=shape, nbpml=nbpml,
                           **(presets[mkey]))

        # Derive timestepping from model spacing
        dt = model.critical_dt * (1.73 if kernel == 'OT4' else 1.0)
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        # Define source geometry (center of domain, just below surface)
        src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
        src.coordinates.data[0, :] = np.array(model.domain_size) * .5
        src.coordinates.data[0, -1] = 30.

        # Define receiver geometry (same as source, but spread across x)
        rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
        rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

        # Create solver object to provide relevant operators
        solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                    kernel=kernel, space_order=space_order)

        # Create adjoint receiver symbol
        srca = Receiver(name='srca', grid=model.grid, time_range=solver.source.time_range,
                        coordinates=solver.source.coordinates.data)

        # Run forward and adjoint operators
        rec, _, _ = solver.forward(save=False)
        solver.adjoint(rec=rec, srca=srca)

        # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
        term1 = np.dot(srca.data.reshape(-1), solver.source.data)
        term2 = linalg.norm(rec.data) ** 2
        info('<Ax,y>: %f, <x, A^Ty>: %f, difference: %4.4e, ratio: %f'
             % (term1, term2, (term1 - term2)/term1, term1 / term2))
        assert np.isclose((term1 - term2)/term1, 0., rtol=1.e-10)

    @pytest.mark.parametrize('space_order', [4, 8, 12])
    @pytest.mark.parametrize('shape', [(60,), (60, 70), (40, 50, 30)])
    def test_adjoint_J(self, shape, space_order):
        """
        Adjoint test for the FWI Jacobian operator.
        The Jacobian operator J generates a linearized shot record (measurements)
        from a model perturbation dm while the adjoint of J generates the FWI gradient
        from an adjoint source (usually data residual). This test uses the conventional
        dot test:
        < Jx, y> = <x ,J^T y>
        """
        t0 = 0.0  # Start time
        tn = 500.  # Final time
        nrec = shape[0]  # Number of receivers
        nbpml = 10 + space_order / 2
        spacing = [10. for _ in shape]

        # Create two-layer "true" model from preset with a fault 1/3 way down
        model = demo_model('layers-isotropic', ratio=3, vp_top=1.5, vp_bottom=2.5,
                           spacing=spacing, space_order=space_order, shape=shape,
                           nbpml=nbpml, dtype=np.float64)

        # Derive timestepping from model spacing
        dt = model.critical_dt
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        # Define source geometry (center of domain, just below surface)
        src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
        src.coordinates.data[0, :] = np.array(model.domain_size) * .5
        src.coordinates.data[0, -1] = 30.

        # Define receiver geometry (same as source, but spread across x)
        rec = Receiver(name='nrec', grid=model.grid, time_range=time_range, npoint=nrec)
        rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

        # Create solver object to provide relevant operators
        solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                    kernel='OT2', space_order=space_order)

        # Create initial model (m0) with a constant velocity throughout
        model0 = demo_model('layers-isotropic', ratio=3, vp_top=1.5, vp_bottom=1.5,
                            spacing=spacing, space_order=space_order, shape=shape,
                            nbpml=nbpml, dtype=np.float64)

        # Compute the full wavefield u0
        _, u0, _ = solver.forward(save=True, m=model0.m)

        # Compute initial born perturbation from m - m0
        dm = (model.m.data - model0.m.data)

        du, _, _, _ = solver.born(dm, m=model0.m)

        # Compute gradientfrom initial perturbation
        im, _ = solver.gradient(du, u0, m=model0.m)

        # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
        term1 = np.dot(im.data.reshape(-1), dm.reshape(-1))
        term2 = linalg.norm(du.data)**2
        info('<Jx,y>: %f, <x, J^Ty>: %f, difference: %4.4e, ratio: %f'
             % (term1, term2, (term1 - term2)/term1, term1 / term2))
        assert np.isclose((term1 - term2)/term1, 0., rtol=1.e-10)

    @pytest.mark.parametrize('shape, coords', [
        ((11, 11), [(.05, .9), (.01, .8)]),
        ((11, 11, 11), [(.05, .9), (.01, .8), (0.07, 0.84)])
    ])
    def test_adjoint_inject_interpolate(self, shape, coords, npoints=19):
        """
        Verify that p.inject is the adjoint of p.interpolate for a
        devito SparseFunction p
        """
        a = unit_box(shape=shape)
        a.data[:] = 0.
        c = unit_box(shape=shape, name='c')
        c.data[:] = 27.
        # Inject receiver
        p = points(a.grid, ranges=coords, npoints=npoints)
        p.data[:] = 1.2
        expr = p.inject(field=a, expr=p)
        # Read receiver
        p2 = points(a.grid, name='points2', ranges=coords, npoints=npoints)
        expr2 = p2.interpolate(expr=c)
        Operator(expr + expr2)(a=a, c=c)
        # < P x, y > - < x, P^T y>
        # Px => p2
        # y => p
        # x => c
        # P^T y => a
        term1 = np.dot(p2.data.reshape(-1), p.data.reshape(-1))
        term2 = np.dot(c.data.reshape(-1), a.data.reshape(-1))
        assert np.isclose((term1-term2) / term1, 0., atol=1.e-6)

import functools
import pickle

import numpy as np
import pytest

from devito import (  # noqa
    CurvilinearGrid, CurvilinearMap, Eq, Function, Grid, Inc, Operator, SparseFunction,
    SparseTimeFunction, SubDomain, TensorFunction, TensorTimeFunction, TerrainFollowing,
    TimeFunction, VectorFunction, VectorTimeFunction, cos, div, grad, inner, resample, sin
)
from devito.mpi import MPI
from examples.seismic import SeismicModel
from examples.seismic.elastic import traction_free_update
from examples.seismic.elastic.sbp import sbp_matrix, sbp_norm

# A smooth surface over a 1000 m line, in m, positive down
_AMPLITUDE = 60.
_WAVELENGTH = 1000.


def sine_surface(x):
    return 20. + _AMPLITUDE*np.sin(2*np.pi*x/_WAVELENGTH)


def sine_surface_dx(x):
    return _AMPLITUDE*2*np.pi/_WAVELENGTH*np.cos(2*np.pi*x/_WAVELENGTH)


def sine_surface_dx2(x):
    return -_AMPLITUDE*(2*np.pi/_WAVELENGTH)**2*np.sin(2*np.pi*x/_WAVELENGTH)


def sine_grid(n, space_order=8, dtype=np.float32):
    return CurvilinearGrid(shape=(n, n), extent=(1000., 800.), space_order=space_order,
                           mapping=TerrainFollowing(surface=sine_surface),
                           dtype=dtype)


def flat_grid(shape, extent, space_order=8):
    surface = np.zeros(shape[:-1])
    return CurvilinearGrid(shape=shape, extent=extent, space_order=space_order,
                           mapping=TerrainFollowing(surface=surface))


def node_coordinates(grid, halo=0):
    """Computational node coordinates (fp64) of `grid`, `halo` points included."""
    vectors = [o + h*np.arange(-halo, n + halo)
               for o, h, n in zip(np.float64(grid.origin), np.float64(grid.spacing),
                                  grid.shape, strict=True)]
    return np.meshgrid(*vectors, indexing='ij')


class TestGrid:

    @pytest.mark.parametrize('shape, extent', [((11, 21), (100., 200.)),
                                               ((7, 9, 11), (60., 80., 100.))])
    def test_flat_is_cartesian(self, shape, extent):
        """
        A flat TerrainFollowing map is the identity: unit and zero metrics, unit
        Jacobian, and node coordinates those of the Cartesian Grid, halo included.
        """
        grid = flat_grid(shape, extent)
        cartesian = Grid(shape=shape, extent=extent)
        assert grid.dimensions == cartesian.dimensions
        assert grid.spacing == cartesian.spacing
        assert grid.min_spacing == min(cartesian.spacing)

        last = grid.dimensions[-1]
        for (physical, computational), metric in grid.metrics.items():
            expected = int(physical is computational)
            if computational is last:
                assert metric.is_Function
                assert np.all(metric.data_with_halo == expected)
            else:
                assert metric == expected
        for metric in grid.metric_derivatives.values():
            assert not metric.is_Function or np.all(metric.data_with_halo == 0)
        assert np.all(grid.jacobian.data_with_halo == 1)

        nodes = node_coordinates(grid, halo=grid.space_order)
        for coordinate, expected in zip(grid.coordinates, nodes, strict=True):
            assert np.allclose(coordinate.data_with_halo, expected, rtol=0, atol=1e-4)

    @pytest.mark.parametrize('space_order', [4, 8])
    def test_metrics_analytic(self, space_order):
        """
        The metric terms match the analytic map, halo included, to the accuracy
        of the finite-difference surface derivatives.
        """
        grid = sine_grid(51, space_order=space_order)
        x, y = grid.dimensions
        xs, eta = node_coordinates(grid, halo=space_order)
        depth, extent = 800., 800.

        surface = sine_surface(xs)
        height = depth - surface
        expected = {
            (x, y): sine_surface_dx(xs)*(eta - extent)/height,
            (y, y): extent/height,
        }
        expected_derivatives = {
            (x, y): (eta - extent)*(sine_surface_dx2(xs)/height +
                                    2*sine_surface_dx(xs)**2/height**2),
        }
        tolerance = {4: 1e-4, 8: 1e-6}[space_order]
        for key, value in expected.items():
            error = np.abs(grid.metrics[key].data_with_halo - value).max()
            assert error < tolerance*np.abs(value).max() + 1e-6
        for key, value in expected_derivatives.items():
            error = np.abs(grid.metric_derivatives[key].data_with_halo - value).max()
            assert error < 10*tolerance*np.abs(value).max()
        assert grid.metrics[(x, x)] == 1 and grid.metrics[(y, x)] == 0
        assert grid.metric_derivatives[(y, y)] == 0
        assert np.allclose(grid.jacobian.data_with_halo, height/extent, rtol=1e-6)
        assert np.allclose(grid.coordinates[1].data_with_halo,
                           surface + eta*height/extent, rtol=1e-6)

    def test_fd_metrics_match_closed_form(self):
        """
        The generic metrics of a CurvilinearMap, finite differences of the
        forward map, agree with the closed forms of TerrainFollowing.
        """
        grid = sine_grid(41)
        mapping = grid.mapping
        vectors = grid._local_vectors

        for generic, closed in [(CurvilinearMap.metrics, mapping.metrics),
                                (CurvilinearMap.metric_derivatives,
                                 mapping.metric_derivatives)]:
            fd_values = generic(mapping, grid, vectors)
            closed_values = closed(grid, vectors)
            for key, value in fd_values.items():
                expected = np.broadcast_to(closed_values[key], value.shape)
                assert np.allclose(value, expected, rtol=0, atol=1e-9)

        jacobian = CurvilinearMap.jacobian(mapping, grid, vectors)
        assert np.allclose(jacobian, mapping.jacobian(grid, vectors), rtol=1e-12)

    def test_array_surface(self):
        """
        A surface given by its node values gives the metrics of the same surface
        given as a callable, in the domain.
        """
        n = 41
        grid = sine_grid(n)
        xs = np.linspace(0., 1000., n)
        mapping = TerrainFollowing(surface=sine_surface(xs))
        sampled = CurvilinearGrid(shape=(n, n), extent=(1000., 800.), mapping=mapping)
        radius = grid.space_order // 2

        for key, metric in grid.metrics.items():
            if metric.is_Function:
                interior = slice(radius, -radius)
                assert np.allclose(sampled.metrics[key].data[interior],
                                   metric.data[interior], rtol=1e-6, atol=1e-7)
        assert np.allclose(sampled.coordinates[1].data, grid.coordinates[1].data,
                           rtol=1e-6)

    def test_inverse(self):
        grid = sine_grid(41)
        points = np.array([[0., 30.], [123.4, 567.8], [999., 800.], [512., -10.]])
        computational = grid.to_computational(points)
        assert np.allclose(computational[:, 0], points[:, 0])
        assert np.allclose(grid.to_physical(computational), points, rtol=0, atol=1e-9)

        # The surface maps to the first row and the bottom to the last
        nodes = np.array([[x, sine_surface(x)] for x in (0., 250., 700.)])
        assert np.allclose(grid.to_computational(nodes)[:, 1], 0, atol=1e-9)
        nodes[:, 1] = 800.
        assert np.allclose(grid.to_computational(nodes)[:, 1], 800., atol=1e-9)

    def test_min_spacing(self):
        """The smallest physical cell size of a tilted plane over a deeper bottom."""
        slope, bottom = 0.1, 400.
        grid = CurvilinearGrid(shape=(21, 41), extent=(200., bottom),
                               mapping=TerrainFollowing(surface=lambda x: slope*x))
        # Columns are shortest where the surface is deepest, at x = 200
        expected = (bottom - slope*200.)/40
        assert np.isclose(grid.min_spacing, expected, rtol=1e-6)

    def test_invalid(self):
        with pytest.raises(ValueError):
            CurvilinearGrid(shape=(5, 5), mapping=TerrainFollowing(np.zeros(4)))
        with pytest.raises(TypeError):
            CurvilinearGrid(shape=(5, 5), mapping=None)
        with pytest.raises(ValueError):
            CurvilinearGrid(shape=(5, 5), mapping=TerrainFollowing(np.zeros(5)),
                            space_order=3)
        # A surface at or below the bottom of the grid, here at depth 1
        with pytest.raises(ValueError, match='above the bottom'):
            CurvilinearGrid(shape=(5, 5), mapping=TerrainFollowing(lambda x: 2*x))
        with pytest.raises(ValueError, match='finite surface'):
            CurvilinearGrid(shape=(5, 5),
                            mapping=TerrainFollowing(np.array([0., 0., np.nan, 0., 0.])))
        # The metric terms are at the nodes and the half points only
        grid = sine_grid(11)
        x, y = grid.dimensions
        with pytest.raises(ValueError, match='nodes and the half points'):
            grid.terms_at({x: x + x.spacing/4})

    def test_function(self):
        """Functions on a CurvilinearGrid have the computational shape and spacing."""
        grid = sine_grid(21)
        u = Function(name='u', grid=grid, space_order=4)
        assert u.shape == (21, 21)
        assert u.grid is grid
        assert grid.is_Curvilinear and not Grid(shape=(4, 4)).is_Curvilinear

    def test_pickle(self):
        """
        A pickled grid rebuilds its coordinate and metric Functions from the map,
        the staggered ones included.
        """
        grid = sine_grid(21, dtype=np.float64)
        jacobian = grid.staggered_terms(grid.dimensions[:1])[1]
        expected = np.array(jacobian.data)

        copy = pickle.loads(pickle.dumps(grid))
        assert copy.shape == grid.shape
        assert np.array_equal(np.array(copy.staggered_terms(copy.dimensions[:1])[1].data),
                              expected)
        assert np.array_equal(np.array(copy.jacobian.data), np.array(grid.jacobian.data))


class TestGridMPI:

    @pytest.mark.parallel(mode=[2, 4])
    def test_metrics(self, mode):
        """The metric Functions of a distributed grid match the serial ones."""
        grid = sine_grid(31)
        x, y = grid.dimensions
        xs, eta = node_coordinates(grid)
        height = 800. - sine_surface(xs)

        expected = sine_surface_dx(xs)*(eta - 800.)/height
        metric = grid.metrics[(x, y)]
        local = np.asarray(metric.data_ro_domain._local)
        assert np.allclose(local, expected[metric.local_indices], rtol=1e-5, atol=1e-6)
        jacobian = grid.jacobian
        assert np.allclose(np.asarray(jacobian.data_ro_domain._local),
                           (height/800.)[jacobian.local_indices], rtol=1e-6)
        assert np.isclose(grid.min_spacing, sine_grid(31).min_spacing)


# Analytic fields for derivative tests, over the physical coordinates
_KX, _KZ = 2*np.pi/400., 2*np.pi/500.


def wave(x, z):
    return np.sin(_KX*x)*np.cos(_KZ*z)


def wave_dx(x, z):
    return _KX*np.cos(_KX*x)*np.cos(_KZ*z)


def wave_dz(x, z):
    return -_KZ*np.sin(_KX*x)*np.sin(_KZ*z)


def wave_dxdz(x, z):
    return -_KX*_KZ*np.cos(_KX*x)*np.sin(_KZ*z)


def wave_laplace(x, z):
    return -(_KX**2 + _KZ**2)*wave(x, z)


def buoyancy(x, z):
    return 1 + 0.5*np.sin(x/300.)*np.cos(z/200.)


def buoyancy_div_grad(x, z):
    """Analytic `div(b*grad(u))` for `b = buoyancy` and `u = wave`."""
    bx = 0.5/300*np.cos(x/300.)*np.cos(z/200.)
    bz = -0.5/200*np.sin(x/300.)*np.sin(z/200.)
    return bx*wave_dx(x, z) + bz*wave_dz(x, z) + buoyancy(x, z)*wave_laplace(x, z)


def physical_function(name, grid, space_order, values):
    """A Function holding `values(x, z)` at the physical nodes, halo included."""
    function = Function(name=name, grid=grid, space_order=space_order)
    coordinates = [c.data_with_halo for c in grid.coordinates]
    function.data_with_halo[:] = values(*coordinates)
    return function


class TestDerivatives:

    @pytest.mark.parametrize('expr', [
        'u.dx', 'u.dy', 'u.dx2', 'u.dy2', 'u.laplace', 'u.dxdy', 'u.dx.T', 'u.dy.T',
        'u.dx2.T', '(b*u.dx).dx + (b*u.dy).dy', '(b*u.dx.T).dx.T',
        'div(b*grad(u))', 'u.dxl', 'u.dyr', 'u.dy(x0=y + y.spacing/2)'
    ])
    def test_flat_is_cartesian(self, expr):
        """
        On a flat TerrainFollowing map, physical derivatives are those of a
        Cartesian Grid, to float32 roundoff.
        """
        shape, extent = (31, 41), (300., 400.)
        rng = np.random.default_rng(0)
        u_data = rng.standard_normal(shape)
        b_data = 1 + rng.random(shape)

        results = []
        for grid in [Grid(shape=shape, extent=extent), flat_grid(shape, extent)]:
            x, y = grid.dimensions
            u = Function(name='u', grid=grid, space_order=8)
            b = Function(name='b', grid=grid, space_order=8)
            out = Function(name='out', grid=grid, space_order=8)
            u.data[:] = u_data
            b.data[:] = b_data
            Operator(Eq(out, eval(expr))).apply()
            results.append(out.data.copy())

        cartesian, curvilinear = results
        assert np.allclose(curvilinear, cartesian, rtol=0,
                           atol=1e-6*np.abs(cartesian).max())

    def test_computational(self):
        """A computational derivative ignores the map."""
        shape, extent = (21, 21), (1000., 800.)
        rng = np.random.default_rng(1)
        u_data = rng.standard_normal(shape)

        results = []
        for grid in [Grid(shape=shape, extent=extent), sine_grid(21)]:
            u = Function(name='u', grid=grid, space_order=4)
            out = Function(name='out', grid=grid, space_order=4)
            u.data[:] = u_data
            expr = u.dx(computational=True) + u.dy2(computational=True)
            Operator(Eq(out, expr)).apply()
            results.append(out.data.copy())
        assert np.array_equal(*results)

    @pytest.mark.parametrize('space_order', [4, 8])
    @pytest.mark.parametrize('expr, solution', [
        ('u.dx', wave_dx),
        ('u.dy', wave_dz),
        ('u.dxdy', wave_dxdz),
        ('u.laplace', wave_laplace),
        ('(b*u.dx).dx + (b*u.dy).dy', buoyancy_div_grad),
        ('div(b*grad(u))', buoyancy_div_grad),
        ('u.dx(x0=x + x.spacing/2).dx(x0=x - x.spacing/2) + '
         'u.dy(x0=y + y.spacing/2).dy(x0=y - y.spacing/2)', wave_laplace),
    ])
    def test_convergence(self, space_order, expr, solution):
        """
        Physical derivatives of analytic fields under a sine surface converge at
        the order of the stencils.
        """
        errors = []
        # Coarse enough for the error to stay above the floor near 1e-9 that the
        # Laplacian on a Cartesian Grid reaches as well
        sizes = {4: [31, 61, 121], 8: [21, 31, 41]}[space_order]
        for n in sizes:
            grid = CurvilinearGrid(shape=(n, n), extent=(1000., 800.),
                                   mapping=TerrainFollowing(sine_surface),
                                   space_order=space_order, dtype=np.float64)
            x, y = grid.dimensions  # noqa
            u = physical_function('u', grid, space_order, wave)  # noqa
            b = physical_function('b', grid, space_order, buoyancy)  # noqa
            out = Function(name='out', grid=grid, space_order=space_order)
            Operator(Eq(out, eval(expr))).apply()

            x, z = (c.data for c in grid.coordinates)
            errors.append(np.abs(out.data - solution(x, z)).max())

        spacing = 1./(np.array(sizes) - 1)
        orders = np.log(np.array(errors[:-1])/np.array(errors[1:])) / \
            np.log(spacing[:-1]/spacing[1:])
        # The finest pair is in the asymptotic regime
        assert orders[-1] > space_order - 1, orders

    @pytest.mark.parametrize('expr, adjoint', [
        ('f.dx', 'g.dx.T'),
        ('f.dy', 'g.dy.T'),
        ('f.dx2', 'g.dx2.T'),
        ('f.dy2', 'g.dy2.T'),
        ('f.dxdy', 'g.dxdy.T'),
        ('f.laplace', 'g.dx2.T + g.dy2.T'),
        ('(b*f.dx).dx', '(b*g.dx.T).dx.T'),
        ('(b*f.dy).dy', '(b*g.dy.T).dy.T'),
        ('f.dy(x0=y + y.spacing/2)', 'g.dy(x0=y + y.spacing/2).T'),
        ('f.dxl', 'g.dxl.T'),
    ])
    def test_adjoint(self, expr, adjoint):
        """
        Dot test `<D f, g> = <f, D.T g>` of physical derivatives under a sine
        surface, the transpose being exact.
        """
        grid = sine_grid(41, dtype=np.float64)
        x, y = grid.dimensions
        rng = np.random.default_rng(2)
        f = Function(name='f', grid=grid, space_order=8)
        g = Function(name='g', grid=grid, space_order=8)
        b = Function(name='b', grid=grid, space_order=8)
        f_out = Function(name='f_out', grid=grid, space_order=8)
        g_out = Function(name='g_out', grid=grid, space_order=8)
        f.data[:] = rng.standard_normal(grid.shape)
        g.data[:] = rng.standard_normal(grid.shape)
        b.data[:] = 1 + rng.random(grid.shape)

        Operator([Eq(f_out, eval(expr)), Eq(g_out, eval(adjoint))]).apply()

        forward = np.dot(f_out.data.ravel(), g.data.ravel())
        backward = np.dot(f.data.ravel(), g_out.data.ravel())
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)

    def test_linear_in_locations(self):
        """
        A derivative of terms whose derivatives are evaluated at different points
        is the sum of the derivatives of the terms, as in a rotated operator.
        """
        grid = sine_grid(41, dtype=np.float64)
        x, y = grid.dimensions
        rng = np.random.default_rng(5)
        f, b = (Function(name=n, grid=grid, space_order=8) for n in 'fb')
        f.data[:] = rng.standard_normal(grid.shape)
        b.data[:] = 1 + rng.random(grid.shape)
        dx = f.dx(x0=x + x.spacing/2)
        dy = f.dy(x0=y + y.spacing/2)
        whole, split = (Function(name=n, grid=grid, space_order=8)
                        for n in ('whole', 'split'))
        Operator([Eq(whole, (b*(dx + dy) + f*(dx + dy)).dx),
                  Eq(split, (b*dx).dx + (b*dy).dx + (f*dx).dx + (f*dy).dx)]).apply()
        assert np.abs(whole.data).max() > 0
        assert np.allclose(whole.data, split.data, rtol=0, atol=1e-12)

    @pytest.mark.parametrize('expr', [
        '(b*u.dx).dx + (b*u.dy).dy',
        '(b._subs(x, x + x.spacing/2)*u.dx(x0=x + x.spacing/2)).dx(x0=x - x.spacing/2)'
        ' + (b._subs(y, y + y.spacing/2)*u.dy(x0=y + y.spacing/2))'
        '.dy(x0=y - y.spacing/2)',
        'u.dx.dy + u.dy.dx',
    ])
    def test_symmetric(self, expr):
        """
        Nested derivatives are symmetric in the inner product weighted by the
        Jacobian, `<J D f, g> = <f, J D g>`, under a sine surface: the outer
        derivative is in conservation form, minus the adjoint of the inner one.
        """
        grid = sine_grid(41, dtype=np.float64)
        x, y = grid.dimensions
        rng = np.random.default_rng(3)
        b = Function(name='b', grid=grid, space_order=8)
        b.data[:] = 1 + rng.random(grid.shape)
        fields, results = [], []
        for _ in range(2):
            u = Function(name='u', grid=grid, space_order=8)
            out = Function(name='out', grid=grid, space_order=8)
            u.data[:] = rng.standard_normal(grid.shape)
            u.data[:8] = u.data[-8:] = u.data[:, :8] = u.data[:, -8:] = 0
            Operator(Eq(out, grid.jacobian*eval(expr))).apply()
            fields.append(u.data.copy())
            results.append(out.data.copy())

        forward = np.dot(results[0].ravel(), fields[1].ravel())
        backward = np.dot(fields[0].ravel(), results[1].ravel())
        scale = np.linalg.norm(results[0])*np.linalg.norm(fields[1])
        assert abs(forward - backward) < 1e-10*scale

    def test_div_grad_adjoint(self):
        """
        `div` and `grad` of a staggered velocity-stress pair are minus each
        other's adjoint in the inner product weighted by the Jacobian at each
        staggered point, `<J v, div(tau)> = -<J tau, grad(v)>`, under a sine
        surface: the derivatives bringing a staggered field back to the nodes
        are in conservation form.
        """
        grid = sine_grid(41, dtype=np.float64)
        rng = np.random.default_rng(4)
        v = VectorFunction(name='v', grid=grid, space_order=8)
        tau = TensorFunction(name='tau', grid=grid, space_order=8, symmetric=True)
        for f in [*v, tau[0, 0], tau[0, 1], tau[1, 1]]:
            f.data[:] = rng.standard_normal(f.shape)
            f.data[:8] = f.data[-8:] = f.data[:, :8] = f.data[:, -8:] = 0
        div_tau = VectorFunction(name='div_tau', grid=grid, space_order=8)
        grad_v = TensorFunction(name='grad_v', grid=grid, space_order=8,
                                symmetric=False)
        eqns = [Eq(div_tau[i], div(tau)[i]) for i in range(2)]
        eqns += [Eq(grad_v[i, j], grad(v)[i, j]) for i in range(2) for j in range(2)]
        Operator(eqns).apply()

        def jacobian(f):
            staggered = tuple(d for d in grid.dimensions if f.staggered.get(d, 0))
            return grid.staggered_terms(staggered)[1].data

        forward = sum(np.sum(jacobian(v[i])*v[i].data*div_tau[i].data)
                      for i in range(2))
        backward = sum(np.sum(jacobian(tau[i, j])*tau[i, j].data*grad_v[i, j].data)
                       for i in range(2) for j in range(2))
        assert np.isclose(forward, -backward, rtol=1e-12, atol=0)

    @pytest.mark.parametrize('space_order', [4, 8])
    @pytest.mark.parametrize('deriv, solution', [('dx', wave_dx), ('dy', wave_dz)])
    @pytest.mark.parametrize('method', ['RSFD', 'FD'])
    def test_staggered_convergence(self, space_order, deriv, solution, method):
        """
        Physical derivatives at the cell centers, rotated (RSFD) or not,
        converge at the order of the stencils, the metric terms being
        interpolated to the centers at the grid order.
        """
        errors = []
        sizes = {4: [31, 61, 121], 8: [21, 41, 61]}[space_order]
        for n in sizes:
            grid = CurvilinearGrid(shape=(n, n), extent=(1000., 800.),
                                   mapping=TerrainFollowing(sine_surface),
                                   space_order=space_order, dtype=np.float64)
            x, y = grid.dimensions
            # A centered FD interpolates `u` along the other Dimension at its
            # interpolation order
            u = Function(name='u', grid=grid, space_order=space_order,
                         interp_order=space_order)
            u.data_with_halo[:] = wave(*(c.data_with_halo for c in grid.coordinates))
            out = Function(name='out', grid=grid, space_order=space_order)
            x0 = {x: x + x.spacing/2, y: y + y.spacing/2}
            Operator(Eq(out, getattr(u, deriv)(method=method, x0=x0))).apply()

            # Physical position of the cell centers
            centers = [c + np.float64(h)/2
                       for c, h in zip(node_coordinates(grid), grid.spacing, strict=True)]
            physical = grid.to_physical(np.stack([c.ravel() for c in centers], axis=1))
            expected = solution(*physical.T).reshape(grid.shape)
            interior = (slice(space_order, -space_order),)*2
            errors.append(np.abs(out.data[interior] - expected[interior]).max())

        spacing = 1./(np.array(sizes) - 1)
        orders = np.log(np.array(errors[:-1])/np.array(errors[1:])) / \
            np.log(spacing[:-1]/spacing[1:])
        assert orders[-1] > space_order - 1, orders

    @pytest.mark.parametrize('deriv', ['dx', 'dy'])
    @pytest.mark.parametrize('staggered', [False, True])
    def test_rsfd_adjoint(self, deriv, staggered):
        """Dot test of rotated staggered physical derivatives."""
        grid = sine_grid(41, dtype=np.float64)
        x, y = grid.dimensions
        x0 = {x: x + x.spacing/2, y: y + y.spacing/2} if staggered else {}
        rng = np.random.default_rng(10)
        f = Function(name='f', grid=grid, space_order=8)
        g = Function(name='g', grid=grid, space_order=8)
        f_out = Function(name='f_out', grid=grid, space_order=8)
        g_out = Function(name='g_out', grid=grid, space_order=8)
        f.data[:] = rng.standard_normal(grid.shape)
        g.data[:] = rng.standard_normal(grid.shape)

        Operator([Eq(f_out, getattr(f, deriv)(method='RSFD', x0=x0)),
                  Eq(g_out, getattr(g, deriv)(method='RSFD', x0=x0).T)]).apply()

        forward = np.dot(f_out.data.ravel(), g.data.ravel())
        backward = np.dot(f.data.ravel(), g_out.data.ravel())
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)

    def test_space_order_check(self):
        grid = sine_grid(21, space_order=4)
        u = Function(name='u', grid=grid, space_order=8)
        with pytest.raises(ValueError):
            _ = u.dx.evaluate

    def test_nesting_check(self):
        """
        A derivative nesting more than two derivatives along the physical
        coordinates, beyond the reach of the halo of the metrics, is refused.
        """
        grid = sine_grid(21)
        x, y = grid.dimensions
        u = Function(name='u', grid=grid, space_order=8)
        _ = u.diff(x, 4).evaluate
        with pytest.raises(NotImplementedError, match='nests 3 derivatives'):
            _ = u.diff(x, 5).evaluate


class TestDerivativesMPI:

    @pytest.mark.parallel(mode=[2, 4])
    def test_div_grad(self, mode):
        """
        A nested physical derivative on a distributed grid matches the serial
        one, the metric halos at the rank boundaries included.
        """
        results = []
        for comm in [None, MPI.COMM_SELF]:
            grid = CurvilinearGrid(shape=(41, 41), extent=(1000., 800.),
                                   mapping=TerrainFollowing(sine_surface),
                                   dtype=np.float64, comm=comm)
            x_coord, z_coord = grid.coordinates
            u = Function(name='u', grid=grid, space_order=8)
            b = Function(name='b', grid=grid, space_order=8)
            out = Function(name='out', grid=grid, space_order=8)
            Operator([Eq(u, sin(_KX*x_coord)*cos(_KZ*z_coord)),
                      Eq(b, 1 + x_coord/1000.)]).apply()
            Operator(Eq(out, div(b*grad(u)) + u.dxdy.T)).apply()
            results.append(out)

        distributed, serial = results
        assert np.allclose(np.asarray(distributed.data_ro_domain._local),
                           serial.data[distributed.local_indices], rtol=0, atol=1e-12)


def random_points(npoint, seed=3):
    """Points below the sine surface, in physical coordinates."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(50., 950., npoint)
    zs = sine_surface(xs) + rng.uniform(5., 700., npoint)
    return np.stack([xs, zs], axis=1)


class TestSparse:

    def test_interpolation_convergence(self):
        """Interpolation of an analytic field at physical points is second order."""
        points = random_points(50)
        errors = []
        sizes = [41, 81, 161]
        for n in sizes:
            grid = sine_grid(n, dtype=np.float64)
            u = Function(name='u', grid=grid, space_order=4)
            u.data[:] = wave(*(c.data for c in grid.coordinates))
            rec = SparseFunction(name='rec', grid=grid, npoint=len(points),
                                 coordinates=points)
            Operator(rec.interpolate(u)).apply()
            errors.append(np.abs(rec.data - wave(*points.T)).max())

        orders = np.log2(np.array(errors[:-1])/np.array(errors[1:]))
        assert np.all(orders > 1.7), orders

    def test_adjoint(self):
        """
        Injection of the data divided by the Jacobian is the adjoint of
        interpolation in the inner product weighted by the Jacobian,
        `<P u, d> = sum(J u P.T (d/J))`.
        """
        grid = sine_grid(41, dtype=np.float64)
        points = random_points(20)
        rng = np.random.default_rng(4)
        u = Function(name='u', grid=grid, space_order=4)
        v = Function(name='v', grid=grid, space_order=4)
        rec = SparseFunction(name='rec', grid=grid, npoint=len(points),
                             coordinates=points)
        src = SparseFunction(name='src', grid=grid, npoint=len(points),
                             coordinates=points)
        u.data[:] = rng.standard_normal(grid.shape)
        src.data[:] = rng.standard_normal(len(points))

        Operator([rec.interpolate(u), src.inject(v, src/grid.jacobian)]).apply()

        forward = np.dot(rec.data, src.data)
        backward = np.sum(grid.jacobian.data*u.data*v.data)
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)

    def test_dirac(self):
        """
        A unit Dirac injected divided by the Jacobian integrates to one over the
        physical domain.
        """
        grid = sine_grid(41)
        points = random_points(3)
        v = Function(name='v', grid=grid, space_order=4)
        src = SparseFunction(name='src', grid=grid, npoint=len(points),
                             coordinates=points)
        src.data[:] = 1.
        Operator(src.inject(v, src/grid.jacobian)).apply()

        cell = np.prod(grid.spacing)
        assert np.isclose(np.sum(grid.jacobian.data*v.data)*cell,
                          len(points)*cell, rtol=1e-5)

    def test_flat_is_cartesian(self):
        """On a flat map, injection and interpolation are those of a Grid."""
        shape, extent = (31, 41), (300., 400.)
        rng = np.random.default_rng(5)
        points = rng.uniform(10., 290., (7, 2))
        u_data = rng.standard_normal(shape)

        results = []
        for grid in [Grid(shape=shape, extent=extent), flat_grid(shape, extent)]:
            u = Function(name='u', grid=grid, space_order=4)
            v = Function(name='v', grid=grid, space_order=4)
            rec = SparseTimeFunction(name='rec', grid=grid, npoint=len(points), nt=1,
                                     coordinates=points)
            u.data[:] = u_data
            rec.data[:] = 1.
            Operator([rec.interpolate(u), rec.inject(v, rec)]).apply()
            results.append((rec.data.copy(), v.data.copy()))

        for cartesian, curvilinear in zip(*results, strict=True):
            assert np.array_equal(cartesian, curvilinear)

    def test_coordinates_editable(self):
        """Moving the points between runs of the same Operator moves the samples."""
        grid = sine_grid(81, dtype=np.float64)
        u = Function(name='u', grid=grid, space_order=4)
        u.data[:] = wave(*(c.data for c in grid.coordinates))
        rec = SparseFunction(name='rec', grid=grid, npoint=10,
                             coordinates=random_points(10))
        op = Operator(rec.interpolate(u))

        for seed in (6, 7):
            points = random_points(10, seed=seed)
            rec.coordinates.data[:] = points
            op.apply()
            assert np.allclose(rec.data, wave(*points.T), atol=1e-2)
            assert np.array_equal(rec.coordinates.data, points)

    @pytest.mark.parametrize('staggered', ['x', 'y', '(x, y)'])
    def test_staggered_amplitude(self, staggered):
        """
        A unit Dirac injected divided by the Jacobian into a staggered field has
        the integral of the same Dirac injected into a node field, the Jacobian
        being taken at the points of the field.
        """
        grid = sine_grid(81, dtype=np.float64)
        x, y = grid.dimensions
        staggered = eval(staggered)
        points = random_points(5)
        integrals = []
        for stagger in [None, staggered]:
            v = Function(name='v', grid=grid, space_order=4, staggered=stagger)
            jacobian = Function(name='jac', grid=grid, space_order=4, staggered=stagger)
            src = SparseFunction(name='src', grid=grid, npoint=len(points),
                                 coordinates=points)
            src.data[:] = 1.
            Operator([Eq(jacobian, grid.jacobian),
                      src.inject(v, src/grid.jacobian)]).apply()
            integrals.append(np.sum(jacobian.data*v.data))

        assert np.isclose(integrals[0], len(points), rtol=1e-10)
        assert np.isclose(integrals[1], integrals[0], rtol=1e-10)

    def test_guard(self):
        """
        A guarded expression is only evaluated for points inside the physical
        domain, between the surface and the bottom.
        """
        grid = sine_grid(41)
        xs = np.array([300., 300., 300., 700.])
        zs = np.array([sine_surface(300.) + 50., sine_surface(300.) - 30., 850.,
                       sine_surface(700.) + 5.])
        points = np.stack([xs, zs], axis=1)
        p = SparseFunction(name='p', grid=grid, npoint=4, coordinates=points)
        q = SparseFunction(name='q', grid=grid, npoint=4, coordinates=points)
        p.data[:] = 1.

        guarded, temps = p.guard(Eq(q, p + 1))
        Operator(temps + [guarded]).apply()

        # Above the surface and below the bottom are outside
        assert np.array_equal(q.data, [2., 0., 0., 2.])


class TestSparseMPI:

    @pytest.mark.parallel(mode=[2, 4])
    def test_interpolation_adjoint(self, mode):
        """
        Interpolation of an analytic field and the Jacobian-weighted adjoint test
        of injection, on a distributed grid.
        """
        grid = sine_grid(41, dtype=np.float64)
        points = random_points(20)
        u = Function(name='u', grid=grid, space_order=4)
        v = Function(name='v', grid=grid, space_order=4)
        weighted = Function(name='weighted', grid=grid, space_order=4)
        rec = SparseFunction(name='rec', grid=grid, npoint=len(points),
                             coordinates=points)
        src = SparseFunction(name='src', grid=grid, npoint=len(points),
                             coordinates=points)
        x_coord, z_coord = grid.coordinates
        Operator(Eq(u, sin(_KX*x_coord)*cos(_KZ*z_coord))).apply()
        src.data[:] = np.linspace(-1., 1., len(points))

        Operator([rec.interpolate(u), src.inject(v, src/grid.jacobian),
                  Eq(weighted, grid.jacobian*v)]).apply()

        # Each rank holds a contiguous block of the points, whose coordinates are
        # still the physical ones after the run
        local_points = np.asarray(rec.coordinates.data._local)
        assert np.array_equal(local_points, points[rec.local_indices[0]])

        # The same sample error as in serial, see test_interpolation_convergence
        local_samples = np.asarray(rec.data._local)
        assert np.abs(local_samples - wave(*local_points.T)).max() < 0.03

        forward = grid.comm.allreduce(np.dot(local_samples,
                                             np.asarray(src.data._local)))
        assert np.isclose(forward, inner(u, weighted), rtol=1e-12)


def cartesian_cover(dtype=np.float32):
    """A Cartesian Grid with the lateral nodes of `sine_grid(41)`, deeper and above."""
    return Grid(shape=(41, 101), extent=(1000., 1000.), origin=(0., -100.), dtype=dtype)


class TestResample:

    def test_analytic(self):
        """Resampling a smooth Cartesian field is second-order accurate."""
        errors = []
        for nz in [51, 101, 201]:
            grid = sine_grid(41, dtype=np.float64)
            cartesian_grid = Grid(shape=(41, nz), extent=(1000., 1000.),
                                  origin=(0., -100.), dtype=np.float64)
            cartesian = Function(name='c', grid=cartesian_grid)
            xs, zs = node_coordinates(cartesian_grid)
            cartesian.data[:] = wave(xs, zs)
            resampled = Function(name='f', grid=grid)
            resample(resampled, cartesian)

            x, z = (c.data for c in grid.coordinates)
            errors.append(np.abs(resampled.data - wave(x, z)).max())

        orders = np.log2(np.array(errors[:-1])/np.array(errors[1:]))
        assert np.all(orders > 1.8), orders

    def test_flat_is_copy(self):
        """On a flat map over the same nodes, resampling copies the values."""
        shape, extent = (21, 31), (200., 300.)
        grid = flat_grid(shape, extent)
        cartesian = Function(name='c', grid=Grid(shape=shape, extent=extent))
        cartesian.data[:] = np.random.default_rng(8).standard_normal(shape)
        resampled = Function(name='f', grid=grid)
        resample(resampled, cartesian)
        assert np.array_equal(resampled.data, cartesian.data)

    def test_adjoint(self):
        """Dot test of the resampling against its transpose."""
        grid = sine_grid(41, dtype=np.float64)
        rng = np.random.default_rng(9)
        cartesian = Function(name='c', grid=cartesian_cover(np.float64))
        transposed = Function(name='ct', grid=cartesian.grid)
        curvilinear = Function(name='f', grid=grid)
        resampled = Function(name='fr', grid=grid)
        cartesian.data[:] = rng.standard_normal(cartesian.shape)
        curvilinear.data[:] = rng.standard_normal(grid.shape)

        resample(resampled, cartesian)
        resample(curvilinear, transposed, transpose=True)

        forward = np.sum(resampled.data*curvilinear.data)
        backward = np.sum(cartesian.data*transposed.data)
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)

    def test_invalid(self):
        grid = sine_grid(11)
        other = sine_grid(11)
        cartesian = Grid(shape=(11, 11))
        with pytest.raises(ValueError, match='not a 2D Cartesian Grid'):
            resample(Function(name='f', grid=grid), Function(name='c', grid=other))
        with pytest.raises(ValueError, match='not defined on a CurvilinearGrid'):
            resample(Function(name='f', grid=cartesian),
                     Function(name='c', grid=cartesian))


class TestResampleMPI:

    @pytest.mark.parallel(mode=[2, 4])
    def test_adjoint(self, mode):
        """
        Resampling between grids with different decompositions matches the serial
        result, and passes the dot test.
        """
        grid = sine_grid(41, dtype=np.float64)
        cartesian = Function(name='c', grid=cartesian_cover(np.float64))
        transposed = Function(name='ct', grid=cartesian.grid)
        curvilinear = Function(name='f', grid=grid)
        resampled = Function(name='fr', grid=grid)
        x_coord, z_coord = grid.coordinates
        cartesian.data[:] = wave(*node_coordinates(cartesian.grid))
        Operator(Eq(curvilinear, sin(_KX*x_coord))).apply()

        resample(resampled, cartesian)
        resample(curvilinear, transposed, transpose=True)

        serial_grid = CurvilinearGrid(shape=(41, 41), extent=(1000., 800.),
                                      mapping=grid.mapping, dtype=np.float64,
                                      comm=MPI.COMM_SELF)
        serial = Function(name='fs', grid=serial_grid)
        serial_cartesian = Function(name='cs', grid=Grid(
            shape=(41, 101), extent=(1000., 1000.), origin=(0., -100.),
            dtype=np.float64, comm=MPI.COMM_SELF
        ))
        serial_cartesian.data[:] = wave(*node_coordinates(serial_cartesian.grid))
        resample(serial, serial_cartesian)
        assert np.allclose(np.asarray(resampled.data_ro_domain._local),
                           serial.data[resampled.local_indices], rtol=0, atol=1e-14)

        forward = inner(resampled, curvilinear)
        backward = inner(cartesian, transposed)
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)


# A 3D surface over a 1000 m x 800 m area, and an analytic field
_K3D = (2*np.pi/400., 2*np.pi/450., 2*np.pi/500.)


def surface_3d(x, y):
    return 20. + 40.*np.sin(2*np.pi*x/1000.)*np.cos(2*np.pi*y/800.)


def surface_3d_derivatives(x, y):
    """First and second derivatives of `surface_3d` along x and y."""
    ax, ay = 2*np.pi/1000., 2*np.pi/800.
    sx = 40.*ax*np.cos(ax*x)*np.cos(ay*y)
    sy = -40.*ay*np.sin(ax*x)*np.sin(ay*y)
    sxx = -40.*ax**2*np.sin(ax*x)*np.cos(ay*y)
    syy = -40.*ay**2*np.sin(ax*x)*np.cos(ay*y)
    return sx, sy, sxx, syy


def wave_3d(x, y, z):
    return np.sin(_K3D[0]*x)*np.sin(_K3D[1]*y + .3)*np.cos(_K3D[2]*z)


def wave_3d_gradient(x, y, z):
    kx, ky, kz = _K3D
    return (kx*np.cos(kx*x)*np.sin(ky*y + .3)*np.cos(kz*z),
            ky*np.sin(kx*x)*np.cos(ky*y + .3)*np.cos(kz*z),
            -kz*np.sin(kx*x)*np.sin(ky*y + .3)*np.sin(kz*z))


def wave_3d_laplace(x, y, z):
    return -sum(k**2 for k in _K3D)*wave_3d(x, y, z)


def grid_3d(n, space_order=4, dtype=np.float64, comm=None):
    return CurvilinearGrid(shape=(n, n, n), extent=(1000., 800., 600.),
                           mapping=TerrainFollowing(surface_3d),
                           space_order=space_order, dtype=dtype, comm=comm)


class Test3D:

    def test_metrics_analytic(self):
        """The 3D metric terms match the analytic map, halo included."""
        grid = grid_3d(31, space_order=8)
        x, y, z = grid.dimensions
        xs, ys, eta = node_coordinates(grid, halo=grid.space_order)
        sx, sy, sxx, syy = surface_3d_derivatives(xs, ys)
        height = 600. - surface_3d(xs, ys)
        distance = eta - 600.

        expected = {(x, z): sx*distance/height, (y, z): sy*distance/height,
                    (z, z): 600./height}
        for key, value in expected.items():
            error = np.abs(grid.metrics[key].data_with_halo - value).max()
            assert error < 1e-6*np.abs(value).max()
        expected = {(x, z): distance*(sxx/height + 2*sx**2/height**2),
                    (y, z): distance*(syy/height + 2*sy**2/height**2)}
        for key, value in expected.items():
            error = np.abs(grid.metric_derivatives[key].data_with_halo - value).max()
            assert error < 1e-5*np.abs(value).max()
        assert np.allclose(grid.jacobian.data_with_halo, height/600., rtol=1e-12)
        assert grid.metrics[(x, x)] == 1 and grid.metrics[(z, x)] == 0

    @pytest.mark.parametrize('expr', ['u.dx', 'u.dz', 'u.laplace', 'u.dxdz',
                                      '(b*u.dy).dy', 'u.dy.T', 'div(b*grad(u))'])
    def test_flat_is_cartesian(self, expr):
        """On a flat 3D map, physical derivatives are those of a Grid."""
        shape, extent = (15, 17, 19), (140., 160., 180.)
        rng = np.random.default_rng(12)
        u_data = rng.standard_normal(shape)
        b_data = 1 + rng.random(shape)

        results = []
        for grid in [Grid(shape=shape, extent=extent),
                     flat_grid(shape, extent, space_order=4)]:
            u = Function(name='u', grid=grid, space_order=4)
            b = Function(name='b', grid=grid, space_order=4)
            out = Function(name='out', grid=grid, space_order=4)
            u.data[:] = u_data
            b.data[:] = b_data
            Operator(Eq(out, eval(expr))).apply()
            results.append(out.data.copy())

        cartesian, curvilinear = results
        assert np.allclose(curvilinear, cartesian, rtol=0,
                           atol=1e-6*np.abs(cartesian).max())

    def test_convergence(self):
        """3D physical derivatives converge at the order of the stencils."""
        sizes = [21, 31, 41]
        errors = {i: [] for i in ['dx', 'dy', 'dz', 'laplace']}
        for n in sizes:
            grid = grid_3d(n)
            u = Function(name='u', grid=grid, space_order=4)
            u.data_with_halo[:] = wave_3d(*(c.data_with_halo for c in grid.coordinates))
            outs = {i: Function(name=f'out_{i}', grid=grid, space_order=4)
                    for i in errors}
            Operator([Eq(outs['dx'], u.dx), Eq(outs['dy'], u.dy), Eq(outs['dz'], u.dz),
                      Eq(outs['laplace'], u.laplace)]).apply()

            coordinates = [c.data for c in grid.coordinates]
            solutions = dict(zip(['dx', 'dy', 'dz'], wave_3d_gradient(*coordinates),
                                 strict=True))
            solutions['laplace'] = wave_3d_laplace(*coordinates)
            for i, out in outs.items():
                errors[i].append(np.abs(out.data - solutions[i]).max())

        spacing = 1./(np.array(sizes) - 1)
        for i, error in errors.items():
            orders = np.log(np.array(error[:-1])/np.array(error[1:])) / \
                np.log(spacing[:-1]/spacing[1:])
            assert orders[-1] > 3.5, (i, orders)

    @pytest.mark.parametrize('expr, adjoint', [
        ('f.dx', 'g.dx.T'),
        ('f.dy', 'g.dy.T'),
        ('f.dz', 'g.dz.T'),
        ('f.dz2', 'g.dz2.T'),
        ('f.dxdz', 'g.dxdz.T'),
        ('(b*f.dy).dy', '(b*g.dy.T).dy.T'),
    ])
    def test_adjoint(self, expr, adjoint):
        """Dot tests of 3D physical derivatives."""
        grid = grid_3d(17)
        rng = np.random.default_rng(13)
        f = Function(name='f', grid=grid, space_order=4)
        g = Function(name='g', grid=grid, space_order=4)
        b = Function(name='b', grid=grid, space_order=4)
        f_out = Function(name='f_out', grid=grid, space_order=4)
        g_out = Function(name='g_out', grid=grid, space_order=4)
        f.data[:] = rng.standard_normal(grid.shape)
        g.data[:] = rng.standard_normal(grid.shape)
        b.data[:] = 1 + rng.random(grid.shape)

        Operator([Eq(f_out, eval(expr)), Eq(g_out, eval(adjoint))]).apply()

        forward = np.dot(f_out.data.ravel(), g.data.ravel())
        backward = np.dot(f.data.ravel(), g_out.data.ravel())
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)

    def test_sparse(self):
        """
        3D interpolation at physical points converges at second order, and is
        the Jacobian-weighted adjoint of the injection divided by the Jacobian.
        """
        rng = np.random.default_rng(14)
        xs, ys = rng.uniform(100., 900., 10), rng.uniform(100., 700., 10)
        points = np.stack([xs, ys, surface_3d(xs, ys) + rng.uniform(10., 500., 10)],
                          axis=1)
        errors = []
        for n in [21, 41]:
            grid = grid_3d(n)
            u = Function(name='u', grid=grid, space_order=4)
            v = Function(name='v', grid=grid, space_order=4)
            u.data[:] = wave_3d(*(c.data for c in grid.coordinates))
            rec = SparseFunction(name='rec', grid=grid, npoint=10, coordinates=points)
            src = SparseFunction(name='src', grid=grid, npoint=10, coordinates=points)
            src.data[:] = rng.standard_normal(10)
            Operator([rec.interpolate(u), src.inject(v, src/grid.jacobian)]).apply()
            errors.append(np.abs(rec.data - wave_3d(*points.T)).max())

            forward = np.dot(rec.data, src.data)
            backward = np.sum(grid.jacobian.data*u.data*v.data)
            assert np.isclose(forward, backward, rtol=1e-12, atol=0)
        assert np.log2(errors[0]/errors[1]) > 1.7

    def test_resample(self):
        """3D resampling is accurate and passes the dot test."""
        grid = grid_3d(21)
        cartesian_grid = Grid(shape=(21, 21, 41), extent=(1000., 800., 800.),
                              origin=(0., 0., -100.), dtype=np.float64)
        cartesian = Function(name='c', grid=cartesian_grid)
        transposed = Function(name='ct', grid=cartesian_grid)
        resampled = Function(name='f', grid=grid)
        curvilinear = Function(name='g', grid=grid)
        cartesian.data[:] = wave_3d(*node_coordinates(cartesian_grid))
        curvilinear.data[:] = np.random.default_rng(15).standard_normal(grid.shape)

        resample(resampled, cartesian)
        resample(curvilinear, transposed, transpose=True)

        expected = wave_3d(*(c.data for c in grid.coordinates))
        assert np.abs(resampled.data - expected).max() < 0.02
        forward = np.sum(resampled.data*curvilinear.data)
        backward = np.sum(cartesian.data*transposed.data)
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)

    @pytest.mark.parallel(mode=[2, 4])
    def test_mpi(self, mode):
        """A 3D nested derivative on a distributed grid matches the serial one."""
        results = []
        for comm in [None, MPI.COMM_SELF]:
            grid = grid_3d(21, comm=comm)
            x_coord, y_coord, z_coord = grid.coordinates
            u = Function(name='u', grid=grid, space_order=4)
            out = Function(name='out', grid=grid, space_order=4)
            Operator(Eq(u, sin(_K3D[0]*x_coord)*cos(_K3D[2]*z_coord))).apply()
            Operator(Eq(out, div(grad(u)) + u.dxdz.T)).apply()
            results.append(out)

        distributed, serial = results
        assert np.allclose(np.asarray(distributed.data_ro_domain._local),
                           serial.data[distributed.local_indices], rtol=0, atol=1e-12)


def surface_grid(shape, extent, surface, rows=8, **kwargs):
    """
    A CurvilinearGrid with `shape` and `extent` below its surface, and `rows`
    mirror rows above it.
    """
    spacing = extent[-1]/(shape[-1] - 1)
    return CurvilinearGrid(shape=shape[:-1] + (shape[-1] + rows,),
                           extent=extent[:-1] + (extent[-1] + rows*spacing,),
                           origin=(0.,)*(len(shape) - 1) + (-rows*spacing,),
                           mapping=TerrainFollowing(surface), **kwargs)


def surface_row(grid):
    """Index of the surface row of a grid, at computational depth 0."""
    return int(round(-float(grid.origin[-1])/float(grid.spacing[-1])))


class Rows(SubDomain):

    """`thickness` rows of a grid from the row `start`, across the grid."""

    def __init__(self, start, thickness, grid):
        self.name = f'rows_{start}_{thickness}'
        self.start = start
        self.thickness = thickness
        super().__init__(grid=grid)

    def define(self, dimensions):
        *lateral, vertical = dimensions
        after = self.grid.shape[-1] - self.start - self.thickness
        return {**{d: d for d in lateral}, vertical: ('middle', self.start, after)}


@functools.cache
def rows_subdomain(grid, start, thickness):
    """The Rows of `grid`, shared by the equations writing them."""
    return Rows(start, thickness, grid)


def below_surface(grid):
    """The rows of `grid` below its surface."""
    start = surface_row(grid) + 1
    return rows_subdomain(grid, start, grid.shape[-1] - start)


def odd_mirror(field, rows):
    """
    The free surface of `field`, e.g. `u.forward`: zero on the surface row `s`
    and the odd mirror `u(z) = -u(2 s - z)` into the `rows` rows above it.
    """
    grid = field.function.grid
    z, surface = grid.dimensions[-1], surface_row(grid)
    return [Eq(field, 0, subdomain=rows_subdomain(grid, surface, 1)),
            Eq(field, -field.subs(z, 2*surface - z),
               subdomain=rows_subdomain(grid, surface - rows, rows))]


def odd_fold(field, coefficient, rows):
    """
    The transpose of `odd_mirror` in an adjoint update of `field`, e.g.
    `v.backward`, which leaves `coefficient` times the operator in the rows
    above the surface, in the inner product weighted by the Jacobian: the rows
    above the surface folded onto the rows below, which they mirror, then zero.
    """
    grid = field.function.grid
    z, surface = grid.dimensions[-1], surface_row(grid)
    folded = (grid.jacobian*field/coefficient).subs(z, 2*surface - z)
    return [Inc(field, -coefficient/grid.jacobian*folded,
                subdomain=rows_subdomain(grid, surface + 1, rows)),
            Eq(field, 0, subdomain=rows_subdomain(grid, 0, surface + 1))]


def staggered_laplacian(f, fd_order):
    """The staggered Laplacian of a wave equation."""
    return sum(getattr(getattr(f, f'd{d.name}')(fd_order=fd_order, x0=d + d.spacing/2),
                       f'd{d.name}')(fd_order=fd_order, x0=d - d.spacing/2)
               for d in f.grid.dimensions)


def free_surface_dot(grid, space_order=8, rows=None, fd_order=None):
    """
    The two sides of `<J A u, w> = <J u, A^T w>` below the surface, with `A` the
    staggered Laplacian after `odd_mirror`, and `A^T` the Laplacian of the field
    held at zero above the surface, followed by `odd_fold`.
    """
    surface = surface_row(grid)
    rows = rows or space_order//2
    fd_order = fd_order or space_order//2
    rng = np.random.default_rng(18)
    u = Function(name='u', grid=grid, space_order=space_order)
    w = Function(name='w', grid=grid, space_order=space_order)
    for f in (u, w):
        values = rng.standard_normal(grid.shape)
        # Zero on and above the surface, and away from the edges and the bottom
        values[..., :surface + 1] = 0
        for axis in range(grid.dim):
            values[(slice(None),)*axis + (slice(-space_order, None),)] = 0
            if axis < grid.dim - 1:
                values[(slice(None),)*axis + (slice(0, space_order),)] = 0
        f.data[:] = values
    au = Function(name='au', grid=grid, space_order=space_order)
    aw = Function(name='aw', grid=grid, space_order=space_order)

    Operator(odd_mirror(u, rows) +
             [Eq(au, staggered_laplacian(u, fd_order))]).apply()
    Operator([Eq(aw, staggered_laplacian(w, fd_order))] +
             odd_fold(aw, 1, rows)).apply()

    jacobian = np.array(grid.jacobian.data)
    below = (slice(None),)*(grid.dim - 1) + (slice(surface + 1, None),)
    forward = np.sum((jacobian*au.data*w.data)[below])
    adjoint = np.sum((jacobian*u.data*aw.data)[below])
    return forward, adjoint


class TestFreeSurface:

    @pytest.mark.parametrize('surface', [np.zeros(21), 30.*np.sin(np.arange(21)/3.)])
    def test_odd_mirror(self, surface):
        """
        The mirror is the odd mirror along the grid lines, `u(-k) = -u(k)` in
        computational space, as on a Grid, over a flat and a curved surface.
        """
        grid = surface_grid((21, 21), (200., 200.), surface)
        rng = np.random.default_rng(16)
        u = TimeFunction(name='u', grid=grid, space_order=8)
        u.data[1] = rng.standard_normal(grid.shape)
        below = np.array(u.data[1, :, 9:13])
        Operator(odd_mirror(u.forward, 4)).apply(time_m=0, time_M=0)

        assert np.all(u.data[1, :, 8] == 0)
        assert np.array_equal(u.data[1, :, 4:8], -below[:, ::-1])

    @pytest.mark.parametrize('surface', [lambda x: 0*x + 20., sine_surface])
    def test_adjoint(self, surface):
        """
        `odd_fold` is the exact transpose of an update followed by `odd_mirror`,
        in the inner product weighted by the Jacobian, over a flat and a sine
        surface.
        """
        grid = surface_grid((41, 41), (1000., 800.), surface, dtype=np.float64)
        forward, adjoint = free_surface_dot(grid)
        assert np.isclose(forward, adjoint, rtol=1e-12, atol=0)

    def test_adjoint_propagation(self):
        """
        Over a flat surface, where the mirror is its own transpose, a backward
        propagation with `odd_fold` is one with `odd_mirror`, for as many steps
        as it runs: the fold follows the updates it depends on.
        """
        grid = surface_grid((41, 41), (400., 400.), np.zeros(41), dtype=np.float64)
        rng = np.random.default_rng(19)
        initial = rng.standard_normal((2,) + grid.shape)
        initial[..., :9] = 0
        results = []
        for transpose in [False, True]:
            u = TimeFunction(name='u', grid=grid, space_order=8, time_order=2)
            u.data[1:] = initial
            update = 2*u - u.forward + 10.*staggered_laplacian(u, 4)
            if transpose:
                eqns = [Eq(u.backward, update)] + odd_fold(u.backward, 10., 4)
            else:
                eqns = odd_mirror(u, 4) + [Eq(u.backward, update,
                                              subdomain=below_surface(grid))]
            Operator(eqns).apply(time_m=1, time_M=20)
            results.append(np.array(u.data[1][:, 9:]))
        mirrored, folded = results
        assert np.abs(mirrored).max() > 0
        assert np.allclose(folded, mirrored, rtol=0, atol=1e-10*np.abs(mirrored).max())

    def test_adjoint_wide(self):
        """The transpose is exact with as many mirror rows as the stencil reaches."""
        grid = surface_grid((41, 41), (1000., 800.), sine_surface, dtype=np.float64)
        forward, adjoint = free_surface_dot(grid, rows=8, fd_order=8)
        assert np.isclose(forward, adjoint, rtol=1e-12, atol=0)

    def test_adjoint_3d(self):
        """The transpose is exact under a surface s(x, y)."""
        grid = surface_grid((25, 25, 25), (1000., 800., 600.), surface_3d, rows=4,
                            space_order=4, dtype=np.float64)
        forward, adjoint = free_surface_dot(grid, space_order=4)
        assert np.isclose(forward, adjoint, rtol=1e-12, atol=0)

    @pytest.mark.parallel(mode=[2, 4])
    def test_adjoint_mpi(self, mode):
        """The fold of a distributed grid matches the serial one."""
        results = []
        for comm in [None, MPI.COMM_SELF]:
            grid = surface_grid((41, 41), (1000., 800.), sine_surface,
                                dtype=np.float64, comm=comm)
            x_coord, z_coord = grid.coordinates
            w = Function(name='w', grid=grid, space_order=8)
            out = Function(name='out', grid=grid, space_order=8)
            Operator(Eq(w, sin(x_coord/150.)*sin(z_coord/120.),
                        subdomain=below_surface(grid))).apply()
            Operator([Eq(out, staggered_laplacian(w, 4))] +
                     odd_fold(out, 1, 4)).apply()
            results.append(out)

        distributed, serial = results
        assert np.abs(serial.data).max() > 0
        assert np.allclose(np.asarray(distributed.data_ro_domain._local),
                           serial.data[distributed.local_indices], rtol=0, atol=1e-12)

    @pytest.mark.parallel(mode=[2, 4])
    def test_mpi(self, mode):
        """The mirror rows of a distributed grid match the serial ones."""
        slope, offset = 0.1, 30.
        results = []
        for comm in [None, MPI.COMM_SELF]:
            grid = surface_grid((41, 41), (1000., 800.), lambda x: offset + slope*x,
                                dtype=np.float64, comm=comm)
            x_coord, z_coord = grid.coordinates
            u = TimeFunction(name='u', grid=grid, space_order=8)
            out = Function(name='out', grid=grid, space_order=8)
            below = below_surface(grid)
            # The field, its mirror and the Laplacian reading the mirror rows
            # next to the surface in one Operator, as in a propagation
            Operator([Eq(u.forward, sin(x_coord/150.) *
                         sin((z_coord - offset - slope*x_coord)/120.), subdomain=below)] +
                     odd_mirror(u.forward, 4) +
                     [Eq(out, u.forward.laplace, subdomain=below)]
                     ).apply(time_m=0, time_M=0)
            results.append(out)

        distributed, serial = results
        assert np.allclose(np.asarray(distributed.data_ro_domain._local),
                           serial.data[distributed.local_indices], rtol=0, atol=1e-12)


def ricker(t, f0=10., t0=.12):
    """Ricker wavelet of peak frequency `f0`, delayed by `t0`."""
    a = (np.pi*f0*(t - t0))**2
    return (1 - 2*a)*np.exp(-a)


def free_space_2d(distance, time, amplitude, velocity):
    """
    Analytic field of `m u_tt - laplace(u) = amplitude*ricker(t)*delta(x)` in 2D
    at `distance` from the source, one column per distance.
    """
    tau = distance/velocity
    field = np.zeros((len(time), len(distance)))
    unit = np.linspace(0, 1, 801)
    for j, tj in enumerate(tau):
        valid = time > tj
        theta = np.arccosh(time[valid]/tj)[:, None]*unit[None, :]
        field[valid, j] = np.trapezoid(ricker(time[valid, None] - tj*np.cosh(theta)),
                                       theta, axis=1)
    return amplitude*field/(2*np.pi)


def tilted_plane_misfit(spacing=10., cfl=.1):
    """
    Relative L2 misfit of an acoustic simulation under a tilted free surface to
    the analytic image-source solution, with the source and the receivers at
    grid nodes.
    """
    velocity, slope = 1500., .1
    grid = surface_grid((int(3000/spacing) + 1, int(1500/spacing) + 1),
                        (3000., 1500.), lambda x: slope*x, rows=4)
    dt = cfl*.8*spacing/velocity
    nt = int(np.ceil(1.2/dt)) + 1
    time = np.arange(nt)*dt

    u = TimeFunction(name='u', grid=grid, space_order=8, time_order=2)
    source = grid.to_physical([[1500., 200.]])
    receivers = grid.to_physical(np.stack([np.linspace(1000., 2000., 101),
                                           np.full(101, 100.)], axis=1))
    src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt, coordinates=source)
    rec = SparseTimeFunction(name='rec', grid=grid, npoint=101, nt=nt,
                             coordinates=receivers)
    src.data[:, 0] = ricker(time)

    surface = odd_mirror(u.forward, 4)
    update = Eq(u.forward, 2*u - u.backward + dt**2*velocity**2*u.laplace)
    source_term = src.inject(u.forward, src*dt**2*velocity**2/grid.jacobian)
    Operator([update] + surface + source_term +
             rec.interpolate(u)).apply(dt=dt, time_M=nt - 2)

    image = source[0] - 2*(slope*source[0, 0] - source[0, 1])/(1 + slope**2) * \
        np.array([slope, -1.])
    amplitude = np.prod(np.array(grid.spacing, dtype=np.float64))
    analytic = (
        free_space_2d(np.linalg.norm(receivers - source, axis=1), time, amplitude,
                      velocity) -
        free_space_2d(np.linalg.norm(receivers - image, axis=1), time, amplitude,
                      velocity)
    )
    return np.linalg.norm(rec.data - analytic)/np.linalg.norm(analytic)


class TestFreeSurfaceWave:

    def test_tilted_plane(self):
        """
        Under a tilted free surface, the odd mirror along the grid lines matches
        the analytic image-source solution: the misfit is 1.6e-3 at a spacing of
        10 m and 4.0e-4 at 5 m, second order, against 9.3e-4 and 2.3e-4 for a
        mirror along the normal to the surface.
        """
        assert tilted_plane_misfit() < 2e-3


class TestSeismicModel:

    @staticmethod
    def model(vp=2., **kwargs):
        return SeismicModel(origin=(0., 0.), spacing=(10., 10.), shape=(51, 41),
                            space_order=8, vp=vp, nbl=10, fs=True, **kwargs)

    def test_mirror_rows(self):
        """
        A flat surface gives the nodes and time step of a Cartesian model below
        the surface, with `space_order` mirror rows above it, the `fsdomain`.
        """
        cartesian = self.model()
        model = self.model(surface=np.zeros(51))
        grid = model.grid
        assert grid.is_Curvilinear
        assert grid.shape == cartesian.grid.shape[:-1] + (cartesian.grid.shape[-1] + 8,)
        assert grid.origin[-1] == -80.
        assert surface_row(grid) == 8
        x_coord, z_coord = (np.array(c.data) for c in grid.coordinates)
        cartesian_z = np.array(cartesian.grid.origin[-1]) + \
            10.*np.arange(cartesian.grid.shape[-1])
        assert np.allclose(z_coord[:, 8:], cartesian_z, atol=1e-4)
        assert np.allclose(z_coord[:, :8], -10.*np.arange(8, 0, -1), atol=1e-4)
        assert set(grid.subdomains) == set(cartesian.grid.subdomains)
        assert model.critical_dt == cartesian.critical_dt

    def test_surface(self):
        """The surface is below the mirror rows, continued flat across the layers."""
        x = 10.*np.arange(51)
        surface = -60.*np.exp(-((x - 250.)/80.)**2)
        model = self.model(surface=surface)
        top = np.array(model.grid.coordinates[-1].data[:, 8])
        assert np.allclose(top, np.pad(surface, 10, mode='edge'), atol=1e-4)

    def test_parameters_above(self):
        """The parameters are continued from the surface into the mirror rows."""
        vp = np.linspace(1.5, 3., 41)[None, :]*np.ones((51, 1))
        x = 10.*np.arange(51)
        model = self.model(surface=-60.*np.exp(-((x - 250.)/80.)**2), vp=vp)
        values = np.array(model.vp.data)
        assert np.allclose(values[:, :8], values[:, 8:9])

    def test_critical_dt(self):
        """The time step follows the smallest cell, under the deepest surface."""
        x = 10.*np.arange(51)
        model = self.model(surface=80.*np.exp(-((x - 250.)/80.)**2))
        cartesian = self.model()
        assert model.grid.min_spacing < 10.
        expected = cartesian.critical_dt*model.grid.min_spacing/10.
        assert np.isclose(model.critical_dt, expected, rtol=1e-2)

    def test_update(self):
        """
        A parameter updated with an array over the physical domain is padded as
        it is when the model is built with it, into the mirror rows too.
        """
        x = 10.*np.arange(51)
        surface = -60.*np.exp(-((x - 250.)/80.)**2)
        vp = np.linspace(1.5, 3., 41)[None, :]*np.ones((51, 1))
        built = self.model(surface=surface, vp=vp)
        updated = self.model(surface=surface, vp=np.ones((51, 41)))
        updated.update('vp', vp)
        assert np.array_equal(np.array(updated.vp.data), np.array(built.vp.data))

    def test_invalid(self):
        with pytest.raises(ValueError, match='not both'):
            self.model(surface=np.zeros(51), grid=self.model().grid)
        with pytest.raises(ValueError, match='needs fs=True'):
            SeismicModel(origin=(0., 0.), spacing=(10., 10.), shape=(51, 41),
                         space_order=8, vp=2., nbl=10, surface=np.zeros(51))
        with pytest.raises(ValueError, match='one value per lateral node'):
            self.model(surface=np.zeros(50))


def elastic_hill(x):
    """A 40 m hill over a 400 m line."""
    return -40.*np.exp(-((x - 200.)/80.)**2)


def elastic_fields(grid):
    """Velocity and stress of space order 4 and time order 1."""
    v = VectorTimeFunction(name='v', grid=grid, space_order=4, time_order=1)
    tau = TensorTimeFunction(name='tau', grid=grid, space_order=4, time_order=1)
    return v, tau


class Lean(CurvilinearMap):
    # x = xi + eta/10, z = eta: a map that does not keep the lateral coordinate

    def forward(self, grid, computational):
        xi, eta = computational
        return xi + eta/10, eta

    def inverse(self, grid, points):
        return np.stack([points[:, 0] - points[:, 1]/10, points[:, 1]], axis=1)


class TestElastic:

    vp, vs, rho = 2000., 1150., 2.

    @property
    def parameters(self):
        """Buoyancy and Lame parameters."""
        return (1/self.rho, self.rho*(self.vp**2 - 2*self.vs**2),
                self.rho*self.vs**2)

    def test_sbp_properties(self):
        """The SBP operators sum by parts, and their interpolations are adjoint."""
        n, m = 20, 12
        dp, dm = sbp_matrix('D', '+', n), sbp_matrix('D', '-', n + 4)
        pp, pm = sbp_matrix('P', '+', n), sbp_matrix('P', '-', n + 4)
        hp, hm = np.diag(sbp_norm('+', n)), np.diag(sbp_norm('-', n + 4))
        boundary = np.zeros((m, m))
        boundary[0, 0] = -1
        assert np.allclose((hm @ dm)[:m, :m] + (hp @ dp).T[:m, :m], boundary,
                           rtol=0, atol=1e-14)
        assert np.allclose((hp @ pp)[:m, :m], (hm @ pm).T[:m, :m], rtol=0, atol=1e-14)

    def test_energy(self):
        """
        The semi-discrete system conserves energy under a hill: in the energy
        inner product, the velocity and stress operators are minus each other's
        adjoints, the traction-free surface included.
        """
        grid = surface_grid((41, 31), (400., 300.), elastic_hill, rows=1,
                            dtype=np.float64)
        s, h, bottom = 1, 10., 300.
        buoyancy, lam, mu = self.parameters
        x = h*np.arange(41)
        jacobian = {False: (bottom - elastic_hill(x))/bottom,
                    True: (bottom - elastic_hill(x + h/2))/bottom}

        def weights(lateral_half, vgrid):
            # Quadrature weights of the SBP grid, times the Jacobian
            norm = np.zeros(grid.shape[-1])
            start = s if vgrid == '+' else s - 1
            norm[start:] = sbp_norm(vgrid, grid.shape[-1] - start)
            return np.outer(jacobian[lateral_half], norm)

        rng = np.random.default_rng(0)
        steps = []
        for _ in range(2):
            v, tau = elastic_fields(grid)
            initial = {}
            for (i, j), start in [((0, 0), s), ((1, 1), s), ((0, 1), s - 1)]:
                values = np.zeros(grid.shape)
                values[10:31, start:start + 16] = rng.standard_normal((21, 16))
                tau[i, j].data[0] = values
                initial[(i, j)] = values
            Operator(traction_free_update(v, tau, buoyancy, lam, mu)).apply(
                time_m=0, time_M=0, dt=1.)
            steps.append((initial, [np.array(c.data[1]) for c in v],
                          {k: np.array(tau[k].data[1]) - initial[k] for k in initial}))

        (initial_a, v_a, _), (_, v_b, rate_b) = steps
        kinetic = (np.sum(weights(True, '+')*v_a[0]*v_b[0]) +
                   np.sum(weights(False, '-')*v_a[1]*v_b[1]))/buoyancy
        compliance = np.linalg.inv([[lam + 2*mu, lam], [lam, lam + 2*mu]])
        normal = sum(compliance[a, b]*np.sum(weights(False, '+')*initial_a[(a, a)] *
                                             rate_b[(b, b)])
                     for a in range(2) for b in range(2))
        shear = np.sum(weights(True, '-')*initial_a[(0, 1)]*rate_b[(0, 1)])/mu
        assert np.isclose(kinetic, -(normal + shear), rtol=1e-10, atol=0)

    def test_tilted_plane(self):
        """
        Under a tilted plane, the velocities in the frame of the plane converge to
        those under a flat surface, with the same source and receivers relative to
        it.
        """
        def record(slope, h):
            grid = surface_grid((int(1500/h) + 1, int(750/h) + 1), (1500., 750.),
                                lambda x: slope*x, rows=1, dtype=np.float64)
            v, tau = elastic_fields(grid)
            dt = .25*h*(750. - 1500.*slope)/750./self.vp
            nt = int(round(.6/dt)) + 1
            tangent = np.array([1., slope])/np.hypot(1., slope)
            normal = np.array([-slope, 1.])/np.hypot(1., slope)
            centre = np.array([750., 750.*slope])
            src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt,
                                     coordinates=[centre + 150.*normal])
            src.data[:, 0] = ricker(dt*np.arange(nt), f0=8., t0=.15)
            receivers = (centre + np.linspace(-300., 300., 31)[:, None]*tangent +
                         20.*normal)
            rec = [SparseTimeFunction(name=f'rec{i}', grid=grid, npoint=31, nt=nt,
                                      coordinates=receivers) for i in range(2)]
            Operator(traction_free_update(v, tau, *self.parameters) +
                     src.inject(tau.forward[0, 0], src*dt/grid.jacobian) +
                     src.inject(tau.forward[1, 1], src*dt/grid.jacobian) +
                     rec[0].interpolate(v[0]) + rec[1].interpolate(v[1])
                     ).apply(time_M=nt - 2, dt=dt)
            data = [np.array(r.data) for r in rec]
            local = np.stack([data[0]*tangent[0] + data[1]*tangent[1],
                              data[0]*normal[0] + data[1]*normal[1]])
            common = np.linspace(0., .6, 601)
            return np.array([[np.interp(common, dt*np.arange(nt), trace)
                              for trace in component.T] for component in local])

        errors = []
        for h in [10., 5.]:
            flat, tilted = record(0., h), record(.1, h)
            errors.append(np.linalg.norm(tilted - flat)/np.linalg.norm(flat))
        assert errors[1] < 2e-2
        assert errors[0]/errors[1] > 3

    def test_invalid(self):
        grid = surface_grid((21, 21), (200., 200.), elastic_hill, rows=1)
        v8 = VectorTimeFunction(name='v8', grid=grid, space_order=8, time_order=1)
        _, tau = elastic_fields(grid)
        with pytest.raises(ValueError, match='space order 4'):
            traction_free_update(v8, tau, *self.parameters)
        no_row = surface_grid((21, 21), (200., 200.), elastic_hill, rows=0)
        with pytest.raises(ValueError, match='row above the surface'):
            traction_free_update(*elastic_fields(no_row), *self.parameters)
        with pytest.raises(ValueError, match='not defined on'):
            traction_free_update(elastic_fields(grid)[0], elastic_fields(no_row)[1],
                                 *self.parameters)
        lean = CurvilinearGrid(shape=(21, 21), extent=(200., 200.), mapping=Lean())
        with pytest.raises(ValueError, match='keeps the lateral'):
            traction_free_update(*elastic_fields(lean), *self.parameters)


class TestElasticMPI:

    @pytest.mark.parallel(mode=[2, 4])
    def test_mpi(self, mode):
        """A distributed elastic propagation under a hill matches the serial one."""
        results = []
        for comm in [None, MPI.COMM_SELF]:
            grid = surface_grid((41, 31), (400., 300.), elastic_hill, rows=1,
                                dtype=np.float64, comm=comm)
            v, tau = elastic_fields(grid)
            # A smooth initial stress next to the surface, in grid indices
            xi, eta = np.meshgrid(np.arange(41), np.arange(32), indexing='ij')
            gauss = np.exp(-((xi - 20.)**2 + (eta - 3.)**2)/3.**2)
            for i in range(2):
                tau[i, i].data[0] = gauss
            Operator(traction_free_update(v, tau, .5, 2.7e6, 2.6e6)).apply(
                time_M=39, dt=1e-3)
            results.append((v, tau))

        (v_d, tau_d), (v_s, tau_s) = results
        pairs = [(v_d[0], v_s[0]), (v_d[1], v_s[1]), (tau_d[0, 0], tau_s[0, 0]),
                 (tau_d[0, 1], tau_s[0, 1])]
        for distributed, serial in pairs:
            scale = np.abs(serial.data[0]).max()
            assert np.allclose(np.asarray(distributed.data_ro_domain[0]._local),
                               serial.data[0][distributed.local_indices[1:]],
                               rtol=0, atol=1e-12*scale)

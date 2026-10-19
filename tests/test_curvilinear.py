import numpy as np
import pytest

from devito import (  # noqa
    CurvilinearGrid, CurvilinearMap, Eq, Function, Grid, Operator, SparseFunction,
    SparseTimeFunction, TensorFunction, TerrainFollowing, TimeFunction, VectorFunction,
    cos, div, grad, inner, sin
)
from devito.mpi import MPI
from examples.seismic import SeismicModel

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

    def test_function(self):
        """Functions on a CurvilinearGrid have the computational shape and spacing."""
        grid = sine_grid(21)
        u = Function(name='u', grid=grid, space_order=4)
        assert u.shape == (21, 21)
        assert u.grid is grid
        assert grid.is_Curvilinear and not Grid(shape=(4, 4)).is_Curvilinear


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
            return grid._staggered_terms(staggered)[1].data

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
        Injection is the adjoint of interpolation in the inner product weighted
        by the Jacobian, `<P u, d> = sum(J u P.T d)`.
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

        Operator([rec.interpolate(u), src.inject(v, src)]).apply()

        forward = np.dot(rec.data, src.data)
        backward = np.sum(grid.jacobian.data*u.data*v.data)
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)

    def test_dirac(self):
        """An injected unit Dirac integrates to one over the physical domain."""
        grid = sine_grid(41)
        points = random_points(3)
        v = Function(name='v', grid=grid, space_order=4)
        src = SparseFunction(name='src', grid=grid, npoint=len(points),
                             coordinates=points)
        src.data[:] = 1.
        Operator(src.inject(v, src)).apply()

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
    def test_staggered_adjoint(self, staggered):
        """
        On a staggered field, injection is the adjoint of interpolation in the
        inner product weighted by the Jacobian at the staggered points.
        """
        grid = sine_grid(41, dtype=np.float64)
        x, y = grid.dimensions
        staggered = eval(staggered)
        points = random_points(20)
        rng = np.random.default_rng(11)
        u = Function(name='u', grid=grid, space_order=4, staggered=staggered)
        v = Function(name='v', grid=grid, space_order=4, staggered=staggered)
        jacobian = Function(name='jac', grid=grid, space_order=4, staggered=staggered)
        rec = SparseFunction(name='rec', grid=grid, npoint=len(points),
                             coordinates=points)
        src = SparseFunction(name='src', grid=grid, npoint=len(points),
                             coordinates=points)
        u.data[:] = rng.standard_normal(grid.shape)
        src.data[:] = rng.standard_normal(len(points))

        Operator([rec.interpolate(u), src.inject(v, src),
                  Eq(jacobian, grid.jacobian)]).apply()

        forward = np.dot(rec.data, src.data)
        backward = np.sum(jacobian.data*u.data*v.data)
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)

    @pytest.mark.parametrize('staggered', ['x', 'y', '(x, y)'])
    def test_staggered_amplitude(self, staggered):
        """
        A unit Dirac injected into a staggered field has the integral of the
        same Dirac injected into a node field, and a smooth staggered field is
        interpolated as accurately as a node one.
        """
        grid = sine_grid(81, dtype=np.float64)
        x, y = grid.dimensions
        staggered = eval(staggered)
        points = random_points(5)
        integrals, errors = [], []
        for stagger in [None, staggered]:
            v = Function(name='v', grid=grid, space_order=4, staggered=stagger)
            u = Function(name='u', grid=grid, space_order=4, staggered=stagger)
            jacobian = Function(name='jac', grid=grid, space_order=4, staggered=stagger)
            src = SparseFunction(name='src', grid=grid, npoint=len(points),
                                 coordinates=points)
            rec = SparseFunction(name='rec', grid=grid, npoint=len(points),
                                 coordinates=points)
            src.data[:] = 1.
            x_coord, z_coord = grid.coordinates
            Operator([Eq(u, sin(_KX*x_coord)*cos(_KZ*z_coord)),
                      Eq(jacobian, grid.jacobian)]).apply()
            Operator([src.inject(v, src), rec.interpolate(u)]).apply()
            integrals.append(np.sum(jacobian.data*v.data))
            errors.append(np.abs(rec.data - wave(*points.T)).max())

        assert np.isclose(integrals[0], len(points), rtol=1e-10)
        assert np.isclose(integrals[1], integrals[0], rtol=1e-10)
        assert errors[1] < 2*errors[0] + 1e-6

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

        Operator([rec.interpolate(u), src.inject(v, src),
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
            grid.resample(resampled, cartesian)

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
        grid.resample(resampled, cartesian)
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

        grid.resample(resampled, cartesian)
        grid.resample(curvilinear, transposed, transpose=True)

        forward = np.sum(resampled.data*curvilinear.data)
        backward = np.sum(cartesian.data*transposed.data)
        assert np.isclose(forward, backward, rtol=1e-12, atol=0)

    def test_invalid(self):
        grid = sine_grid(11)
        other = sine_grid(11)
        with pytest.raises(ValueError):
            grid.resample(Function(name='f', grid=grid), Function(name='c', grid=other))
        with pytest.raises(ValueError):
            grid.resample(Function(name='f', grid=other),
                          Function(name='c', grid=Grid(shape=(11, 11))))


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

        grid.resample(resampled, cartesian)
        grid.resample(curvilinear, transposed, transpose=True)

        serial_grid = CurvilinearGrid(shape=(41, 41), extent=(1000., 800.),
                                      mapping=grid.mapping, dtype=np.float64,
                                      comm=MPI.COMM_SELF)
        serial = Function(name='fs', grid=serial_grid)
        serial_cartesian = Function(name='cs', grid=Grid(
            shape=(41, 101), extent=(1000., 1000.), origin=(0., -100.),
            dtype=np.float64, comm=MPI.COMM_SELF
        ))
        serial_cartesian.data[:] = wave(*node_coordinates(serial_cartesian.grid))
        serial_grid.resample(serial, serial_cartesian)
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
        the Jacobian-weighted adjoint of injection.
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
            Operator([rec.interpolate(u), src.inject(v, src)]).apply()
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

        grid.resample(resampled, cartesian)
        grid.resample(curvilinear, transposed, transpose=True)

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


def plane_reflection(points, slope, offset):
    """Mirror image of `points` through the plane `z = offset + slope . lateral`."""
    slope = np.atleast_1d(slope)
    norm = np.sqrt(1 + slope @ slope)
    normal = np.append(slope, -1.)/norm
    distance = (points[:, :-1] @ slope - points[:, -1] + offset)/norm
    return points - 2*distance[:, None]*normal[None, :]


def smooth_field(points):
    return np.sin(points[:, 0]/150.)*np.cos(points[:, -1]/120.) + 0.3*points[:, -1]/100.


def odd_field(points, slope, offset):
    """A smooth field, odd w.r.t. the plane: zero on it, exact image across it."""
    return smooth_field(points) - smooth_field(plane_reflection(points, slope, offset))


def mirrored_ghosts(grid, space_order, slope, offset):
    """
    Ghost rows set by `grid.free_surface` from the exact odd field below the
    surface, the exact odd field on the ghost rows, and the ghost rows of the
    mirror along the grid columns.
    """
    # The field and the coordinates share the halo of the grid space order
    u = TimeFunction(name='u', grid=grid, space_order=space_order)
    coordinates = np.stack([c.data_with_halo.ravel() for c in grid.coordinates],
                           axis=1)
    exact = odd_field(coordinates, slope, offset).reshape(
        grid.coordinates[0].data_with_halo.shape
    )
    u.data_with_halo[1] = 0.
    u.data_with_halo[1][..., space_order:] = exact[..., space_order:]

    Operator(grid.free_surface(u.forward)).apply(time_m=0, time_M=0)

    rows = slice(space_order - space_order//2, space_order)
    # Mirror along the grid columns, u(-k) = -u(k)
    mirrored_rows = slice(space_order + space_order//2, space_order, -1)
    # Away from the lateral edges, where the image stencils read the halo
    lateral = (slice(2*space_order, -2*space_order),)*(grid.dim - 1)
    return (u.data_with_halo[1][lateral + (rows,)], exact[lateral + (rows,)],
            -exact[lateral + (mirrored_rows,)])


class TestFreeSurface:

    def test_flat_is_odd_mirror(self):
        """Over a flat surface, the mirror is the usual `u(-k) = -u(k)`."""
        grid = flat_grid((21, 21), (200., 200.))
        x, y = grid.dimensions
        t = grid.stepping_dim
        rng = np.random.default_rng(16)
        u = TimeFunction(name='u', grid=grid, space_order=8)
        u.data[1] = rng.standard_normal(grid.shape)
        reference = TimeFunction(name='reference', grid=grid, space_order=8)
        reference.data[1] = u.data[1]

        mirror = [Eq(reference[t+1, x, -k], -reference[t+1, x, k]) for k in range(1, 5)]
        Operator(grid.free_surface(u.forward) + mirror).apply(time_m=0, time_M=0)

        assert np.array_equal(u.data_with_halo[1, 8:-8, 4:8],
                              reference.data_with_halo[1, 8:-8, 4:8])
        assert np.all(u.data[1, :, 0] == 0)

    def test_fields_share_weights(self):
        """
        Fields mirrored in the same Operator share the interpolation weights of
        their images, and are mirrored alike.
        """
        grid = sine_grid(31)
        rng = np.random.default_rng(17)
        u = TimeFunction(name='u', grid=grid, space_order=8)
        v = TimeFunction(name='v', grid=grid, space_order=8)
        u.data[1] = rng.standard_normal(grid.shape)
        v.data[1] = u.data[1]

        eqns = grid.free_surface(u.forward) + grid.free_surface(v.forward)
        weights = {f for e in eqns for f in e.rhs.free_symbols
                   if getattr(f, 'function', None) is not None and
                   f.function.name.startswith('image')}
        assert len({f.function for f in weights}) == 1
        Operator(eqns).apply(time_m=0, time_M=0)
        # To float32 roundoff, the compiler ordering the two loops differently
        assert np.allclose(u.data_with_halo[1], v.data_with_halo[1], rtol=0,
                           atol=1e-6)

    @pytest.mark.parametrize('space_order', [4, 8])
    def test_tilted_plane(self, space_order):
        """
        Over a tilted plane, the ghost rows are the exact image of an odd field,
        to the order of the interpolation, while mirroring along the grid
        columns is only second order.
        """
        slope, offset = 0.1, 30.
        errors, column_errors = [], []
        sizes = {4: [41, 81], 8: [21, 41]}[space_order]
        for n in sizes:
            grid = CurvilinearGrid(shape=(n, n), extent=(1000., 800.),
                                   mapping=TerrainFollowing(lambda x: offset + slope*x),
                                   space_order=space_order, dtype=np.float64)
            ghosts, exact, column = mirrored_ghosts(grid, space_order, slope, offset)
            errors.append(np.abs(ghosts - exact).max())
            column_errors.append(np.abs(column - exact).max())

        assert np.log2(errors[0]/errors[1]) > space_order - 1
        assert np.log2(column_errors[0]/column_errors[1]) < 2.5
        assert errors[1] < 1e-3*column_errors[1]

    def test_tilted_plane_3d(self):
        """The image mirror over a tilted plane in 3D."""
        slope, offset = np.array([0.1, -0.05]), 50.
        errors = []
        for n in [21, 41]:
            grid = CurvilinearGrid(shape=(n, n, n), extent=(1000., 800., 600.),
                                   mapping=TerrainFollowing(
                                       lambda x, y: offset + slope[0]*x + slope[1]*y),
                                   space_order=4, dtype=np.float64)
            ghosts, exact, _ = mirrored_ghosts(grid, 4, slope, offset)
            errors.append(np.abs(ghosts - exact).max())
        assert np.log2(errors[0]/errors[1]) > 3

    @pytest.mark.parallel(mode=[2, 4])
    def test_mpi(self, mode):
        """The ghost rows of a distributed grid match the serial ones."""
        slope, offset = 0.1, 30.
        results = []
        for comm in [None, MPI.COMM_SELF]:
            grid = CurvilinearGrid(shape=(41, 41), extent=(1000., 800.),
                                   mapping=TerrainFollowing(lambda x: offset + slope*x),
                                   space_order=8, dtype=np.float64, comm=comm)
            x_coord, z_coord = grid.coordinates
            u = TimeFunction(name='u', grid=grid, space_order=8)
            out = Function(name='out', grid=grid, space_order=8)
            Operator(Eq(u.forward, sin(x_coord/150.) *
                        sin((z_coord - offset - slope*x_coord)/120.))
                     ).apply(time_m=0, time_M=0)
            # The Laplacian reads the ghost rows next to the surface
            Operator(grid.free_surface(u.forward) +
                     [Eq(out, u.forward.laplace)]).apply(time_m=0, time_M=0)
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


def tilted_plane_misfit(mirror, spacing=10., cfl=.1):
    """
    Relative L2 misfit of an acoustic simulation under a tilted free surface to
    the analytic image-source solution, with the source and the receivers at
    grid nodes.
    """
    velocity, slope = 1500., .1
    grid = CurvilinearGrid(shape=(int(3000/spacing) + 1, int(1500/spacing) + 1),
                           extent=(3000., 1500.),
                           mapping=TerrainFollowing(lambda x: slope*x))
    x, y = grid.dimensions
    t = grid.stepping_dim
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

    if mirror == 'image':
        surface = grid.free_surface(u.forward)
    else:
        surface = [Eq(u[t+1, x, 0], 0)]
        surface += [Eq(u[t+1, x, -k], -u[t+1, x, k]) for k in range(1, 5)]
    update = Eq(u.forward, 2*u - u.backward + dt**2*velocity**2*u.laplace)
    Operator([update] + surface + src.inject(u.forward, src*dt**2*velocity**2) +
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
        Under a tilted free surface, the image mirror matches the analytic
        image-source solution more closely than the mirror along the columns.
        """
        image = tilted_plane_misfit('image')
        column = tilted_plane_misfit('column')
        assert image < 1.5e-3
        assert column > 1.4*image


class TestSeismicModel:

    @staticmethod
    def model(**kwargs):
        return SeismicModel(origin=(0., 0.), spacing=(10., 10.), shape=(51, 41),
                            space_order=8, vp=2., nbl=10, fs=True, **kwargs)

    def test_flat_is_cartesian(self):
        """A flat surface gives the grid, subdomains and time step of a Grid."""
        cartesian = self.model()
        model = self.model(surface=np.zeros(51))
        assert model.grid.is_Curvilinear
        assert model.grid.shape == cartesian.grid.shape
        assert model.grid.origin == cartesian.grid.origin
        assert model.grid.extent == cartesian.grid.extent
        assert set(model.grid.subdomains) == set(cartesian.grid.subdomains)
        assert model.critical_dt == cartesian.critical_dt

    def test_surface(self):
        """The first row is the surface, continued flat across the absorbing layers."""
        x = 10.*np.arange(51)
        surface = -60.*np.exp(-((x - 250.)/80.)**2)
        model = self.model(surface=surface)
        top = np.array(model.grid.coordinates[-1].data[:, 0])
        assert np.allclose(top, np.pad(surface, 10, mode='edge'), atol=1e-4)

    def test_critical_dt(self):
        """The time step follows the smallest cell, under the deepest surface."""
        x = 10.*np.arange(51)
        model = self.model(surface=80.*np.exp(-((x - 250.)/80.)**2))
        cartesian = self.model()
        assert model.grid.min_spacing < 10.
        expected = cartesian.critical_dt*model.grid.min_spacing/10.
        assert np.isclose(model.critical_dt, expected, rtol=1e-2)

    def test_invalid(self):
        with pytest.raises(ValueError, match='needs fs=True'):
            SeismicModel(origin=(0., 0.), spacing=(10., 10.), shape=(51, 41),
                         space_order=8, vp=2., nbl=10, surface=np.zeros(51))
        with pytest.raises(ValueError, match='one value per lateral node'):
            self.model(surface=np.zeros(50))

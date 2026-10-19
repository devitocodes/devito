import numpy as np
import pytest

from devito import (  # noqa
    CurvilinearGrid, CurvilinearMap, Eq, Function, Grid, Operator, SparseFunction,
    SparseTimeFunction, TerrainFollowing, cos, div, grad, inner, sin
)
from devito.mpi import MPI

# A smooth surface over a 1000 m line, in m, positive down
_AMPLITUDE = 60.
_WAVELENGTH = 1000.


def sine_surface(x):
    return 20. + _AMPLITUDE*np.sin(2*np.pi*x/_WAVELENGTH)


def sine_surface_dx(x):
    return _AMPLITUDE*2*np.pi/_WAVELENGTH*np.cos(2*np.pi*x/_WAVELENGTH)


def sine_surface_dx2(x):
    return -_AMPLITUDE*(2*np.pi/_WAVELENGTH)**2*np.sin(2*np.pi*x/_WAVELENGTH)


def sine_grid(n, space_order=8, depth=800., dtype=np.float32):
    return CurvilinearGrid(shape=(n, n), extent=(1000., 800.), space_order=space_order,
                           mapping=TerrainFollowing(surface=sine_surface, depth=depth),
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
        mapping = TerrainFollowing(surface=sine_surface(xs), depth=800.)
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
        slope, depth = 0.1, 400.
        grid = CurvilinearGrid(shape=(21, 41), extent=(200., 400.),
                               mapping=TerrainFollowing(surface=lambda x: slope*x,
                                                        depth=depth))
        # Columns are shortest where the surface is deepest, at x = 200
        expected = (depth - slope*200.)/40
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
                                   mapping=TerrainFollowing(sine_surface, depth=800.),
                                   space_order=space_order, dtype=np.float64)
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
                                   mapping=TerrainFollowing(sine_surface, depth=800.),
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

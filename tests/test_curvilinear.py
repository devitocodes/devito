import numpy as np
import pytest

from devito import CurvilinearGrid, CurvilinearMap, Function, Grid, TerrainFollowing

# A smooth surface over a 1000 m line, in m, positive down
_AMPLITUDE = 60.
_WAVELENGTH = 1000.


def sine_surface(x):
    return 20. + _AMPLITUDE*np.sin(2*np.pi*x/_WAVELENGTH)


def sine_surface_dx(x):
    return _AMPLITUDE*2*np.pi/_WAVELENGTH*np.cos(2*np.pi*x/_WAVELENGTH)


def sine_surface_dx2(x):
    return -_AMPLITUDE*(2*np.pi/_WAVELENGTH)**2*np.sin(2*np.pi*x/_WAVELENGTH)


def sine_grid(n, space_order=8, depth=800.):
    return CurvilinearGrid(shape=(n, n), extent=(1000., 800.), space_order=space_order,
                           mapping=TerrainFollowing(surface=sine_surface, depth=depth))


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

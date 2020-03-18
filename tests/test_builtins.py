import pytest
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import misc

from conftest import skipif
from devito import Grid, Function, TimeFunction, switchconfig
from devito.builtins import assign, norm, gaussian_smooth, initialize_function, inner
from devito.data import LEFT, RIGHT
from devito.tools import as_tuple
from devito.types import SubDomain, SparseTimeFunction

pytestmark = skipif(['yask', 'ops'])


class TestAssign(object):

    """
    Class for testing the assign builtin.
    """

    def test_single_scalar(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)

        assign(f, 4)

        assert np.all(f.data == 4)

    def test_multiple_fns_single_scalar(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)
        functions = [f, g, h]
        assign(functions, 2)

        assert np.all(f.data == 2)
        assert np.all(g.data == 2)
        assert np.all(h.data == 2)

    def test_multiple_fns_multiple_scalar(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)
        functions = [f, g, h]
        scalars = [1, 2, 3]
        assign(functions, scalars)

        assert np.all(f.data == 1)
        assert np.all(g.data == 2)
        assert np.all(h.data == 3)

    def test_equations_with_options(self):

        class CompDomain(SubDomain):

            name = 'comp_domain'

            def define(self, dimensions):
                return {d: ('middle', 1, 1) for d in dimensions}

        comp_domain = CompDomain()
        grid = Grid(shape=(4, 4), subdomains=comp_domain)

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        functions = [f, g]
        scalars = 2
        options = [None, {'subdomain': grid.subdomains['comp_domain']}]
        assign(functions, scalars, options=options)

        assert np.all(f.data == 2)
        assert np.all(np.array(g.data) == [[0, 0, 0, 0],
                                           [0, 2, 2, 0],
                                           [0, 2, 2, 0],
                                           [0, 0, 0, 0]])

    @skipif('nompi')
    @pytest.mark.parallel(mode=4)
    def test_assign_parallel(self):
        a = np.arange(64).reshape((8, 8))
        grid = Grid(shape=a.shape)

        f = Function(name='f', grid=grid, dtype=np.int32)
        f.data[:] = a
        g = Function(name='g', grid=grid, dtype=np.int32)
        assign(g, f)

        loc_shape = np.array(grid._distributor.shape)
        loc_coords = np.array(grid._distributor.mycoords)
        start = loc_shape*loc_coords
        stop = loc_shape*(loc_coords+1)

        slices = []
        for i, j in zip(start, stop):
            slices.append(slice(i, j, 1))
        slices = as_tuple(slices)
        assert np.all(a[slices] - np.array(g.data[:]) == 0)


class TestGaussianSmooth(object):

    """
    Class for testing the Gaussian smooth builtin.
    """

    @pytest.mark.parametrize('sigma', [1, 2, 3, 4, 5])
    def test_gs_1d_int(self, sigma):
        """Test the Gaussian smoother in 1d."""

        a = np.arange(970, step=5)
        sp_smoothed = gaussian_filter(a, sigma=sigma)
        dv_smoothed = gaussian_smooth(a, sigma=sigma)

        assert np.amax(np.abs(sp_smoothed - np.array(dv_smoothed))) <= 1

    @pytest.mark.parametrize('sigma', [1, 2])
    def test_gs_1d_float(self, sigma):
        """Test the Gaussian smoother in 1d on array of float."""

        a = np.array([1.2, 2.7, 3.9, 4.1, 5.2, 6.5, 7.1, 9.3, 11.0])
        sp_smoothed = gaussian_filter(a, sigma=sigma)
        dv_smoothed = gaussian_smooth(a, sigma=sigma)

        assert np.amax(np.abs(sp_smoothed - np.array(dv_smoothed))) <= 1e-5

    @pytest.mark.parametrize('sigma', [(1, 1), 2, (1, 3), (5, 5)])
    def test_gs_2d_int(self, sigma):
        """Test the Gaussian smoother in 2d."""

        a = misc.ascent()
        sp_smoothed = gaussian_filter(a, sigma=sigma)
        dv_smoothed = gaussian_smooth(a, sigma=sigma)

        try:
            s = max(sigma)
        except TypeError:
            s = sigma
        assert np.amax(np.abs(sp_smoothed - np.array(dv_smoothed))) <= s

    @pytest.mark.parametrize('sigma', [(1, 1), 2, (1, 3), (5, 5)])
    def test_gs_2d_float(self, sigma):
        """Test the Gaussian smoother in 2d."""

        a = misc.ascent()
        a = a+0.1
        sp_smoothed = gaussian_filter(a, sigma=sigma)
        dv_smoothed = gaussian_smooth(a, sigma=sigma)

        assert np.amax(np.abs(sp_smoothed - np.array(dv_smoothed))) <= 1e-5

    @skipif('nompi')
    @pytest.mark.parallel(mode=4)
    def test_gs_parallel(self):
        a = np.arange(64).reshape((8, 8))
        grid = Grid(shape=a.shape)

        f = Function(name='f', grid=grid, dtype=np.int32)
        f.data[:] = a

        sp_smoothed = gaussian_filter(a, sigma=1)
        dv_smoothed = gaussian_smooth(f, sigma=1)

        loc_shape = np.array(grid._distributor.shape)
        loc_coords = np.array(grid._distributor.mycoords)
        start = loc_shape*loc_coords
        stop = loc_shape*(loc_coords+1)

        slices = []
        for i, j in zip(start, stop):
            slices.append(slice(i, j, 1))
        slices = as_tuple(slices)
        assert np.all(sp_smoothed[slices] - np.array(dv_smoothed.data[:]) == 0)


class TestInitializeFunction(object):

    """
    Class for testing the initialize function builtin.
    """

    def test_if_serial(self):
        """Test in serial."""
        a = np.arange(16).reshape((4, 4))
        grid = Grid(shape=(12, 12))
        f = Function(name='f', grid=grid, dtype=np.int32)
        initialize_function(f, a, 4, mode='reflect')

        assert np.all(a[:, ::-1] - np.array(f.data[4:8, 0:4]) == 0)
        assert np.all(a[:, ::-1] - np.array(f.data[4:8, 8:12]) == 0)
        assert np.all(a[::-1, :] - np.array(f.data[0:4, 4:8]) == 0)
        assert np.all(a[::-1, :] - np.array(f.data[8:12, 4:8]) == 0)

    def test_nbl_zero(self):
        """Test for nbl = 0."""
        a = np.arange(16).reshape((4, 4))
        grid = Grid(shape=(4, 4))
        f = Function(name='f', grid=grid, dtype=np.int32)
        initialize_function(f, a, 0)

        assert np.all(a[:] - np.array(f.data[:]) == 0)

    @skipif('nompi')
    @pytest.mark.parallel(mode=4)
    def test_if_parallel(self):
        a = np.arange(36).reshape((6, 6))
        grid = Grid(shape=(18, 18))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        f = Function(name='f', grid=grid, halo=((3, 3), (3, 3)), dtype=np.int32)
        initialize_function(f, a, 6, mode='reflect')

        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(a[::-1, 0:3] - np.array(f.data[0:6, 6:9]) == 0)
            assert np.all(a[0:3, ::-1] - np.array(f.data[6:9, 0:6]) == 0)
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(a[::-1, 3:6] - np.array(f.data[0:6, 9:12]) == 0)
            assert np.all(a[0:3, ::-1] - np.array(f.data[6:9, 12:18]) == 0)
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(a[::-1, 0:3] - np.array(f.data[12:18, 6:9]) == 0)
            assert np.all(a[3:6, ::-1] - np.array(f.data[9:12, 0:6]) == 0)
        else:
            assert np.all(a[::-1, 3:6] - np.array(f.data[12:18, 9:12]) == 0)
            assert np.all(a[3:6, ::-1] - np.array(f.data[9:12, 12:18]) == 0)


class TestBuiltinsResult(object):

    """
    Test the builtins
    """

    def test_serial_vs_parallel(self):
        grid = Grid(shape=(100, 100))

        f = TimeFunction(name='f', grid=grid)
        f.data[:] = np.arange(10000).reshape((100, 100))

        assert np.isclose(norm(f),
                          switchconfig(openmp=True)(norm)(f),
                          rtol=1e-5)

    def test_inner_sparse(self):
        """
        Test that inner produces the correct result against numpy
        """
        grid = Grid((101, 101), extent=(1000., 1000.))

        nrec = 101
        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = np.linspace(0., 1000., nrec)
        rec_coordinates[:, -1] = 20.

        rec0 = SparseTimeFunction(name='rec0', grid=grid, nt=1001, npoint=nrec,
                                  coordinates=rec_coordinates)
        rec1 = SparseTimeFunction(name='rec1', grid=grid, nt=1001, npoint=nrec,
                                  coordinates=rec_coordinates)

        rec0.data[:, :] = np.random.randn(*rec0.shape)
        rec1.data[:, :] = np.random.randn(*rec1.shape)
        term1 = inner(rec0, rec1)
        term2 = np.dot(rec0.data.reshape(-1), rec1.data.reshape(-1))
        assert np.isclose(term1/term2 - 1, 0.0, rtol=0.0, atol=1e05)

    def test_norm_sparse(self):
        """
        Test that norm produces the correct result against numpy
        """
        grid = Grid((101, 101), extent=(1000., 1000.))

        nrec = 101
        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = np.linspace(0., 1000., nrec)
        rec_coordinates[:, -1] = 20.

        rec0 = SparseTimeFunction(name='rec0', grid=grid, nt=1001, npoint=nrec,
                                  coordinates=rec_coordinates)

        rec0.data[:, :] = np.random.randn(*rec0.shape)
        term1 = np.linalg.norm(rec0.data)
        term2 = norm(rec0)
        assert np.isclose(term1/term2 - 1, 0.0, rtol=0.0, atol=1e05)

import pytest
import numpy as np
from scipy.ndimage import gaussian_filter

from conftest import skipif
from devito import Grid, Function
from devito.builtins import assign, gaussian_smooth
from devito.tools import as_tuple
from devito.types import SubDomain

pytestmark = skipif(['yask', 'ops'])


class TestAssign(object):
    """
    Class for testing the assign builtin
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


class TestGaussianSmooth(object):
    """
    Class for testing the Gaussian smooth builtin
    """
    def test_gs_serial(self):
        """Test in serial."""

        a = np.arange(50, step=2).reshape((5, 5))
        sp_smoothed = gaussian_filter(a, sigma=1)
        dv_smoothed = gaussian_smooth(a, sigma=1)

        assert np.all(sp_smoothed - np.array(dv_smoothed) == 0)

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

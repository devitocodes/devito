import pytest
import numpy as np
from scipy.ndimage import gaussian_filter

from conftest import skipif
from devito import Grid, Function
from devito.builtins import gaussian_smooth
from devito.tools import as_tuple

pytestmark = skipif(['yask', 'ops'])


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

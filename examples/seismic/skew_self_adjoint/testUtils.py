import numpy as np
import pytest
from devito import Grid, Function, configuration
from examples.seismic.skew_self_adjoint import *
configuration['language'] = 'openmp'
configuration['log-level'] = 'ERROR'


class TestUtils(object):

    @pytest.mark.parametrize('shape', [(40, 50), (40, 50, 60)])
    @pytest.mark.parametrize('npad', [10, ])
    @pytest.mark.parametrize('w', [2.0 * np.pi * 0.010, ])
    @pytest.mark.parametrize('qmin', [0.1, 1.0])
    @pytest.mark.parametrize('qmax', [10.0, 100.0])
    @pytest.mark.parametrize('sigma', [None, 11])
    @pytest.mark.parametrize('dtype', [np.float32, ])
    def test_setupWOverQ(self, shape, npad, w, qmin, qmax, sigma, dtype):
        """
        Test for the function that sets up the w/Q attenuation model.
        This is not a correctness test, we just ensure that the output model:
            - value is bounded [w/Qmin, w/Qmax]
            - value is w/Qmin in corners
            - value is w/Qmax in center
        """

        tol = np.finfo(dtype).eps

        # question: do we need to test for float32, float64?
        if len(shape) == 2:
            origin = (0.0, 0.0)
            extent = (1.0, 1.0)
            grid = Grid(extent=extent, shape=shape, origin=origin, dtype=dtype)
            wOverQ = Function(name='wOverQ', grid=grid, space_order=8)
            setup_wOverQ(wOverQ, w, qmin, qmax, npad, sigma=None)
            q = (1 / (wOverQ.data / w))
            assert np.isclose(np.min(q[:]), qmin, 10 * tol)
            assert np.isclose(np.max(q[:]), qmax, 10 * tol)

            nx, nz = q.data.shape
            assert np.isclose(q.data[0, 0], qmin, atol=tol)
            assert np.isclose(q.data[0, nz-1], qmin, atol=tol)
            assert np.isclose(q.data[nx-1, 0], qmin, atol=tol)
            assert np.isclose(q.data[nx-1, nz-1], qmin, atol=tol)

            assert np.isclose(q.data[nx//2, nz//2], qmax, atol=tol)

        else:
            origin = (0.0, 0.0, 0.0)
            extent = (1.0, 1.0, 1.0)
            grid = Grid(extent=extent, shape=shape, origin=origin, dtype=np.float32)
            wOverQ = Function(name='wOverQ', grid=grid, space_order=8)
            setup_wOverQ(wOverQ, w, qmin, qmax, npad, sigma=None)
            q = (1 / (wOverQ.data / w))
            assert np.isclose(np.min(q), qmin, 10 * tol)
            assert np.isclose(np.max(q), qmax, 10 * tol)

            nx, ny, nz = q.data.shape
            assert np.isclose(q.data[0, 0, 0], qmin, atol=tol)
            assert np.isclose(q.data[0, 0, nz-1], qmin, atol=tol)
            assert np.isclose(q.data[0, ny-1, 0], qmin, atol=tol)
            assert np.isclose(q.data[0, ny-1, nz-1], qmin, atol=tol)
            assert np.isclose(q.data[nx-1, 0, 0], qmin, atol=tol)
            assert np.isclose(q.data[nx-1, 0, nz-1], qmin, atol=tol)
            assert np.isclose(q.data[nx-1, ny-1, 0], qmin, atol=tol)
            assert np.isclose(q.data[nx-1, ny-1, nz-1], qmin, atol=tol)

            assert np.isclose(q.data[nx//2, ny//2, nz//2], qmax, atol=tol)

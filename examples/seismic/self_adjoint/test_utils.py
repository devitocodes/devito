import numpy as np
import pytest
from devito import Grid, Function
from examples.seismic.self_adjoint import setup_w_over_q


class TestUtils(object):

    def make_grid(self, shape, dtype):
        origin = tuple([0.0 for s in shape])
        extent = tuple([s - 1 for s in shape])
        return Grid(extent=extent, shape=shape, origin=origin, dtype=dtype)

    @pytest.mark.parametrize('shape', [(41, 51), (41, 51, 61)])
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

        tol = 10 * np.finfo(dtype).eps
        grid = self.make_grid(shape, dtype)
        wOverQ = Function(name='wOverQ', grid=grid)
        setup_w_over_q(wOverQ, w, qmin, qmax, npad, sigma=None)
        q = (1 / (wOverQ.data / w))

        assert np.isclose(np.min(q[:]), qmin, 10 * tol)
        assert np.isclose(np.max(q[:]), qmax, 10 * tol)

        # question: do we need to test for float32, float64?
        if len(shape) == 2:
            nx, nz = q.data.shape
            assert np.isclose(q.data[0, 0], qmin, atol=tol)
            assert np.isclose(q.data[0, nz-1], qmin, atol=tol)
            assert np.isclose(q.data[nx-1, 0], qmin, atol=tol)
            assert np.isclose(q.data[nx-1, nz-1], qmin, atol=tol)
            assert np.isclose(q.data[nx//2, nz//2], qmax, atol=tol)
        else:
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

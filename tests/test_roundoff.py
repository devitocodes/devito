import pytest
import numpy as np

from conftest import skipif
from devito import Grid, Constant, TimeFunction, Eq, Operator, switchconfig


class TestRoundoff:
    """
    Class for checking round-off errors are not unexpectedly creeping in to certain
    stencil types.
    """
    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    @switchconfig(log_level='DEBUG', safe_math=True)
    def test_lm_forward(self, dat, dtype):
        """
        Test logistic map with forward term that should cancel.
        """
        iterations = 10000
        r = Constant(name='r', dtype=dtype)
        r.data = dtype(dat)
        s = dtype(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1), dtype=dtype)
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2, dtype=dtype)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2,
                          dtype=dtype)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.forward-f0.forward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.forward-f1.forward))

        initial_condition = dtype(0.7235)

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, dtype(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1, 3)], f1.data[iterations+1],
                           atol=0, rtol=0)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    @switchconfig(log_level='DEBUG', safe_math=True)
    def test_lm_backward(self, dat, dtype):
        """
        Test logistic map with backward term that should cancel.
        """
        iterations = 10000
        r = Constant(name='r', dtype=dtype)
        r.data = dtype(dat)
        s = dtype(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1), dtype=dtype)
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2, dtype=dtype)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2,
                          dtype=dtype)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.backward-f0.backward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.backward-f1.backward))

        initial_condition = dtype(0.7235)

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, dtype(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1, 3)], f1.data[iterations+1],
                           atol=0, rtol=0)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    @switchconfig(log_level='DEBUG', safe_math=True)
    @skipif('cpu64-arm')
    def test_lm_fb(self, dat, dtype):
        """
        Test logistic map with forward and backward terms that should cancel.
        """
        iterations = 10000
        r = Constant(name='r', dtype=dtype)
        r.data = dtype(dat)
        s = dtype(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1), dtype=dtype)
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2, dtype=dtype)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2,
                          dtype=dtype)

        initial_condition = dtype(0.7235)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.backward
                                     - f0.backward+(1.0/s)*dt*f0.forward-f0.forward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.backward
                                     - f1.backward+(1.0/s)*dt*f1.forward-f1.forward))

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, dtype(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1, 3)], f1.data[iterations+1],
                           atol=0, rtol=0)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    @switchconfig(log_level='DEBUG', safe_math=True)
    @skipif('cpu64-arm')
    def test_lm_ds(self, dat, dtype):
        """
        Test logistic map with 2nd derivative term that should cancel.
        """
        iterations = 10000
        r = Constant(name='r', dtype=dtype)
        r.data = dtype(0.5*dat)
        s = dtype(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1), dtype=dtype)

        f0 = TimeFunction(name='f0', grid=grid, time_order=2, dtype=dtype)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2,
                          dtype=dtype)

        initial_condition = dtype(0.7235)

        lmap0 = Eq(f0.forward, -r*f0.dt2*s**2*(1.0-f0) +
                   r*(1.0-f0)*(f0.backward+f0.forward))
        lmap1 = Eq(f1.forward, -r*f1.dt2*s**2*(1.0-f1) +
                   r*(1.0-f1)*(f1.backward+f1.forward))

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, dtype(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1, 3)], f1.data[iterations+1],
                           atol=0, rtol=0)

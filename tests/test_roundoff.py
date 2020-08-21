import pytest
import numpy as np

from devito import Grid, Constant, TimeFunction, Eq, Operator


class TestRoundoff(object):
    """
    Class for testing SubDomains
    """
    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0], 'dtype', [np.float32, np.float64])
    def test_lm_forward(self, dat, dtype):
        """
        ...
        """
        iterations = 10000
        r= Constant(name='r')
        r.data = dat
        s = 0.01

        grid = Grid(shape=(2, 2), extent=(1, 1))
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2, dtype=dtype)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2, dtype=dtype)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.forward-f0.forward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.forward-f1.forward))

        initial_condition = 0.7235

        f0.data[:, :, :] = initial_condition
        f1.data[:, :, :] = initial_condition

        op0 = Operator(lmap0)
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1,3)], f1.data[iterations+1], atol=0, rtol=0)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    def test_lm_backward(self, dat):
        """
        ...
        """
        iterations = 10000
        r= Constant(name='r')
        r.data = dat
        s = 0.01

        grid = Grid(shape=(2, 2), extent=(1, 1))
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.backward-f0.backward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.backward-f1.backward))

        initial_condition = 0.7235

        f0.data[:, :, :] = initial_condition
        f1.data[:, :, :] = initial_condition

        op0 = Operator(lmap0)
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1,3)], f1.data[iterations+1], atol=0, rtol=0)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    def test_lm_fb(self, dat):
        """
        ...
        """
        iterations = 10000
        r= Constant(name='r')
        r.data = dat
        s = 0.01

        grid = Grid(shape=(2, 2), extent=(1, 1))
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.backward-f0.backward+(1.0/s)*dt*f0.forward-f0.forward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.backward-f1.backward+(1.0/s)*dt*f1.forward-f1.forward))

        initial_condition = 0.7235

        f0.data[:, :, :] = initial_condition
        f1.data[:, :, :] = initial_condition

        op0 = Operator(lmap0)
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1,3)], f1.data[iterations+1], atol=0, rtol=0)

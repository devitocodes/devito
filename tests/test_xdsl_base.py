import numpy as np
import pytest

from devito import Grid, TimeFunction, Eq, XDSLOperator, Operator, solve


def test_xdsl_I():
    # Define a simple Devito Operator
    grid = Grid(shape=(3,))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_xdsl_II():
    # Define a simple Devito Operator
    grid = Grid(shape=(4, 4))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_xdsl_III():
    # Define a simple Devito Operator
    grid = Grid(shape=(5, 5, 5))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_diffusion_2D():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))

    # Devito setup
    f = TimeFunction(name='f', grid=grid, space_order=2)
    f.data[:] = 1
    eqn = Eq(f.dt, 0.5 * f.laplace)
    op = Operator(Eq(f.forward, solve(eqn, f.forward)))
    op.apply(time_M=1, dt=0.1)

    # xDSL-Devito setup
    f2 = TimeFunction(name='f2', grid=grid, space_order=2)
    f2.data[:] = 1
    eqn = Eq(f2.dt, 0.5 * f2.laplace)
    op = XDSLOperator(Eq(f2.forward, solve(eqn, f2.forward)))
    op.apply(time_M=1, dt=0.1)
   
    assert np.isclose(f.data, f2.data, rtol=1e-06).all()


@pytest.mark.parametrize('shape', [(11, 11), (31, 31), (51, 51), (101, 101)])
def test_diffusion_2D_II(shape):
    # Define a simple Devito Operator
    grid = Grid(shape=shape)
    rng = np.random.default_rng(123)
    dx = 2. / (shape[0] - 1)
    dy = 2. / (shape[1] - 1)
    sigma = .1
    dt = sigma * dx * dy
    nt = 10
    
    arr1 = rng.random(shape)

    # Devito setup
    f = TimeFunction(name='f', grid=grid, space_order=2)
    f.data[:] = arr1
    eqn = Eq(f.dt, 0.5 * f.laplace)
    op = Operator(Eq(f.forward, solve(eqn, f.forward)))
    op.apply(time_M=nt, dt=dt)

    # xDSL-Devito setup
    f2 = TimeFunction(name='f2', grid=grid, space_order=2)
    f2.data[:] = arr1
    eqn = Eq(f2.dt, 0.5 * f2.laplace)
    op = XDSLOperator(Eq(f2.forward, solve(eqn, f2.forward)))
    op.apply(time_M=nt, dt=dt)
 
    max_error = np.max(np.abs(f.data - f2.data))
    assert np.isclose(max_error, 0.0, atol=1e-04)
    assert np.isclose(f.data, f2.data, rtol=1e-05).all()


@pytest.mark.parametrize('shape', [(11, 11, 11), (31, 31, 31), (51, 51, 51), (101, 101, 101)])
def test_diffusion_3D_II(shape):
    shape = (10, 10, 10)
    grid = Grid(shape=shape)
    rng = np.random.default_rng(123)
    dx = 2. / (shape[0] - 1)
    dy = 2. / (shape[1] - 1)
    dz = 2. / (shape[2] - 1)

    sigma = .1
    dt = sigma * dx * dy * dz
    nt = 50

    rng = np.random.default_rng(123)
    
    arr1 = rng.random(shape)

    # Devito setup
    f = TimeFunction(name='f', grid=grid, space_order=2)
    f.data[:] = arr1
    eqn = Eq(f.dt, 0.5 * f.laplace)
    op = Operator(Eq(f.forward, solve(eqn, f.forward)))
    op.apply(time_M=nt, dt=dt)

    # xDSL-Devito setup
    f2 = TimeFunction(name='f2', grid=grid, space_order=2)
    f2.data[:] = arr1
    eqn = Eq(f2.dt, 0.5 * f2.laplace)
    op = XDSLOperator(Eq(f2.forward, solve(eqn, f2.forward)))
    op.apply(time_M=50, dt=dt)
 
    max_error = np.max(np.abs(f.data - f2.data))
    assert np.isclose(max_error, 0.0, atol=1e-04)
    assert np.isclose(f.data, f2.data, rtol=1e-05).all()
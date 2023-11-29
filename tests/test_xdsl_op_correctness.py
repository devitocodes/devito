import numpy as np
from devito import Grid, TimeFunction, Eq, Operator, norm
import pytest
# flake8: noqa

def test_udx():

    # Define a simple Devito Operator
    grid = Grid(shape=(5, 5))
    u = TimeFunction(name='u', grid=grid)
    u.data[:] = 0.1
    eq = Eq(u.forward, u.dx)
    op = Operator([eq])
    op.apply(time_M=5)
    norm1 = norm(u)

    u.data[:] = 0.1

    xdsl_op = Operator([eq], opt='xdsl')
    xdsl_op.apply(time_M=5)
    norm2 = norm(u)
    
    assert np.isclose(norm1, norm2,   atol=1e-5, rtol=0)
    assert np.isclose(norm1, 14636.3955, atol=1e-5, rtol=0)

def test_u_plus1_conversion():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    u.data[:] = 0
    eq = Eq(u.forward, u + 1)
    op = Operator([eq])
    op.apply(time_M=5)
    norm1 = norm(u)
    
    u.data[:] = 0
    xdsl_op = Operator([eq], opt='xdsl')
    xdsl_op.apply(time_M=5)
    norm2 = norm(u)

    assert np.isclose(norm1, norm2, atol=1e-5, rtol=0)
    assert np.isclose(norm1, 23.43075, atol=1e-5, rtol=0)

@pytest.mark.xfail(reason="Needs a fix in offsets")
def test_u_and_v_conversion():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid, time_order=2)
    v = TimeFunction(name='v', grid=grid, time_order=2)
    u.data[:] = 0.0001
    v.data[:] = 0.0001
    eq0 = Eq(u.forward, u.dt)
    eq1 = Eq(v.forward, u.dt)
    op = Operator([eq0, eq1])
    op.apply(time_M=5, dt=0.1)
    norm_u = norm(u)
    norm_v = norm(v)

    u.data[:] = 0.0001
    v.data[:] = 0.0001
    xdsl_op = Operator([eq0, eq1], opt='xdsl')
    xdsl_op.apply(time_M=5, dt=0.1)
    norm_u2 = norm(u)
    norm_v2 = norm(v)

    assert np.isclose(norm_u, norm_u2, atol=1e-5, rtol=0)
    assert np.isclose(norm_u, 26.565891, atol=1e-5, rtol=0)
    assert np.isclose(norm_v, norm_v2, atol=1e-5, rtol=0)
    assert np.isclose(norm_v, 292.49646, atol=1e-5, rtol=0)

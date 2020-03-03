import numpy as np
import pytest

from conftest import skipif
from devito import Grid, Eq, Operator, TimeFunction

pytestmark = skipif(['yask', 'ops'])


@pytest.fixture
def grid(shape=(11, 11)):
    return Grid(shape=shape)


@pytest.fixture
def a(grid):
    """Forward time data object, unrolled (save=True)"""
    return TimeFunction(name='a', grid=grid, time_order=1, save=6)


@pytest.fixture
def b(grid):
    """Backward time data object, unrolled (save=True)"""
    return TimeFunction(name='b', grid=grid, time_order=1, save=6)


@pytest.fixture
def c(grid):
    """Forward time data object, buffered (save=False)"""
    return TimeFunction(name='c', grid=grid, time_order=1, save=None)


@pytest.fixture
def d(grid):
    """Forward time data object, unrolled (save=True), end order"""
    return TimeFunction(name='d', grid=grid, time_order=2, save=6)


def test_forward(a):
    a.data[0, :] = 1.
    Operator(Eq(a.forward, a + 1.))()
    for i in range(a.shape[0]):
        assert np.allclose(a.data[i, :], 1. + i, rtol=1.e-12)


def test_backward(b):
    b.data[-1, :] = 7.
    Operator(Eq(b.backward, b - 1.))()
    for i in range(b.shape[0]):
        assert np.allclose(b.data[i, :], 2. + i, rtol=1.e-12)


def test_forward_unroll(a, c, nt=5):
    """Test forward time marching with a buffered and an unrolled t"""
    a.data[0, :] = 1.
    c.data[0, :] = 1.
    eqn_c = Eq(c.forward, c + 1.)
    eqn_a = Eq(a.forward, c.forward)
    Operator([eqn_c, eqn_a])(time=nt-1)
    for i in range(nt):
        assert np.allclose(a.data[i, :], 1. + i, rtol=1.e-12)


def test_forward_backward(a, b, nt=5):
    """Test a forward operator followed by a backward marching one"""
    a.data[0, :] = 1.
    b.data[0, :] = 1.
    eqn_a = Eq(a.forward, a + 1.)
    Operator(eqn_a)(time=nt-1)

    eqn_b = Eq(b, a + 1.)
    Operator(eqn_b)(time=nt-1)
    for i in range(nt):
        assert np.allclose(b.data[i, :], 2. + i, rtol=1.e-12)


def test_forward_backward_overlapping(a, b, nt=5):
    """
    Test a forward operator followed by a backward one, but with
    overlapping operator definitions.
    """
    a.data[0, :] = 1.
    b.data[0, :] = 1.
    op_fwd = Operator(Eq(a.forward, a + 1.))
    op_bwd = Operator(Eq(b, a + 1.))

    op_fwd(time=nt-1)
    op_bwd(time=nt-1)
    for i in range(nt):
        assert np.allclose(b.data[i, :], 2. + i, rtol=1.e-12)


def test_loop_bounds_forward(d):
    """Test the automatic bound detection for forward time loops"""
    d.data[:] = 1.
    eqn = Eq(d, 2. + d.dt2)
    Operator(eqn, dle=None)(dt=1.)
    assert np.allclose(d.data[0, :], 1., rtol=1.e-12)
    assert np.allclose(d.data[-1, :], 1., rtol=1.e-12)
    for i in range(1, d.data.shape[0]-1):
        assert np.allclose(d.data[i, :], 1. + i, rtol=1.e-12)


def test_loop_bounds_backward(d):
    """Test the automatic bound detection for backward time loops"""
    d.data[:] = 5.
    eqn = Eq(d.backward, d - 1)
    op = Operator(eqn, dle=None)
    op()
    assert np.allclose(d.data[0, :], 0., rtol=1.e-12)
    assert np.allclose(d.data[-1, :], 5., rtol=1.e-12)
    for i in range(1, d.data.shape[0]-1):
        assert np.allclose(d.data[i, :], i, rtol=1.e-12)
